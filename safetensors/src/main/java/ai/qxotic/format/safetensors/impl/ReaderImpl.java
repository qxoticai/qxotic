package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.DType;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.SafetensorsFormatException;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.math.BigDecimal;
import java.math.BigInteger;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

class ReaderImpl {
    static final String METADATA_KEY = "__metadata__";
    static final int HEADER_SIZE_BYTES = 8;
    static final int ALIGNMENT_DEFAULT_VALUE = AlignmentSupport.DEFAULT_VALUE;
    static final String ALIGNMENT_KEY = AlignmentSupport.KEY;
    private static final long MAX_SIZE_T = 281474976710655L;
    private static final String DTYPE_FIELD = "dtype";
    private static final String SHAPE_FIELD = "shape";
    private static final String OFFSETS_FIELD = "data_offsets";
    private static final Set<String> TENSOR_FIELDS =
            Set.of(DTYPE_FIELD, SHAPE_FIELD, OFFSETS_FIELD);

    private static final class ParsedHeader {
        final Map<String, String> metadata;
        final Map<String, TensorEntry> tensors;

        ParsedHeader(Map<String, String> metadata, Map<String, TensorEntry> tensors) {
            this.metadata = metadata;
            this.tensors = tensors;
        }
    }

    static Safetensors read(ReadableByteChannel channel) throws IOException {
        // Read header size
        ByteBuffer sizeBuffer =
                ByteBuffer.allocate(HEADER_SIZE_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        if (readFully(channel, sizeBuffer) != HEADER_SIZE_BYTES) {
            throw new IOException("File too small: expected at least 8 bytes for header size");
        }
        sizeBuffer.flip();
        long headerSizeLong = sizeBuffer.getLong();
        if (headerSizeLong <= 0 || headerSizeLong > Integer.MAX_VALUE) {
            throw new SafetensorsFormatException("Invalid header size: " + headerSizeLong);
        }
        int headerSize = Math.toIntExact(headerSizeLong);

        // Read header
        ByteBuffer headerBuffer = ByteBuffer.allocate(headerSize);
        if (readFully(channel, headerBuffer) != headerSize) {
            throw new IOException("Failed to read header: expected " + headerSize + " bytes");
        }
        headerBuffer.flip();

        // Validate header starts with '{'
        if (headerBuffer.get(0) != (byte) '{') {
            throw new SafetensorsFormatException("Invalid header: must start with '{' (0x7B)");
        }

        // Parse JSON header
        String headerJson = StandardCharsets.UTF_8.decode(headerBuffer).toString();
        Map<?, ?> header;
        try {
            header = requireObject(JSON.parse(headerJson), "Header JSON must be an object");
        } catch (JSON.ParseException e) {
            throw new SafetensorsFormatException("Invalid JSON in header: " + e.getMessage(), e);
        }

        ParsedHeader parsedHeader = parseHeader(header);

        long dataOffset = HEADER_SIZE_BYTES + headerSize;

        // Validate no overlaps in byte buffer
        validateTensors(parsedHeader.tensors.values());

        return new SafetensorsImpl(dataOffset, parsedHeader.metadata, parsedHeader.tensors);
    }

    private static ParsedHeader parseHeader(Map<?, ?> header) {
        Map<String, String> metadata = new LinkedHashMap<>();
        Map<String, TensorEntry> tensors = new LinkedHashMap<>();

        for (Map.Entry<?, ?> entry : header.entrySet()) {
            String key = requireString(entry.getKey(), "Header keys must be strings");
            if (METADATA_KEY.equals(key)) {
                parseMetadata(entry.getValue(), metadata);
            } else {
                tensors.put(key, parseTensor(key, entry.getValue()));
            }
        }

        return new ParsedHeader(metadata, tensors);
    }

    private static void parseMetadata(Object metadataObj, Map<String, String> metadata) {
        Map<?, ?> metadataMap = requireObject(metadataObj, "__metadata__ must be an object");
        for (Map.Entry<?, ?> entry : metadataMap.entrySet()) {
            String key = requireString(entry.getKey(), "__metadata__ keys must be strings");
            String value =
                    requireString(
                            entry.getValue(),
                            "__metadata__ values must be strings for key: " + key);
            metadata.put(key, value);
        }
        AlignmentSupport.validateHeaderMetadata(metadata);
    }

    private static TensorEntry parseTensor(String tensorName, Object tensorObj) {
        Map<?, ?> tensorMap =
                requireObject(
                        tensorObj, "Tensor entry must be an object " + tensorContext(tensorName));

        validateTensorFields(tensorName, tensorMap);

        DType dtype = parseDType(tensorName, tensorMap.get(DTYPE_FIELD));

        long[] shape = parseShape(tensorName, tensorMap.get(SHAPE_FIELD));
        long[] offsets = parseDataOffsets(tensorName, tensorMap.get(OFFSETS_FIELD));
        long begin = offsets[0];
        long end = offsets[1];
        validateOffsets(tensorName, begin, end);

        long size = end - begin;
        long expectedSize = dtype.byteSizeForShape(shape);
        if (expectedSize != size) {
            throw new SafetensorsFormatException(
                    "Size mismatch "
                            + tensorContext(tensorName)
                            + ": expected "
                            + expectedSize
                            + " bytes, got "
                            + size);
        }

        return TensorEntry.create(tensorName, dtype, shape, begin);
    }

    private static void validateOffsets(String tensorName, long begin, long end) {
        if (begin < 0 || end < 0 || begin > end) {
            throw new SafetensorsFormatException("Invalid offsets " + tensorContext(tensorName));
        }
    }

    private static void validateTensorFields(String tensorName, Map<?, ?> tensorMap) {
        if (!tensorMap.keySet().containsAll(TENSOR_FIELDS)) {
            throw new SafetensorsFormatException(
                    "Missing required tensor fields " + tensorContext(tensorName));
        }
        if (tensorMap.size() != TENSOR_FIELDS.size()) {
            throw new SafetensorsFormatException("Unknown fields " + tensorContext(tensorName));
        }
    }

    private static DType parseDType(String tensorName, Object dtypeObj) {
        String dtypeString =
                requireString(dtypeObj, "dtype must be a string " + tensorContext(tensorName));
        try {
            return DType.valueOf(dtypeString);
        } catch (IllegalArgumentException e) {
            throw new SafetensorsFormatException(
                    "Invalid or unsupported dtype '"
                            + dtypeString
                            + "' "
                            + tensorContext(tensorName));
        }
    }

    private static long[] parseShape(String tensorName, Object shapeObj) {
        return parseLongArray(shapeObj, SHAPE_FIELD, tensorName, -1);
    }

    private static long[] parseDataOffsets(String tensorName, Object offsetsObj) {
        return parseLongArray(offsetsObj, OFFSETS_FIELD, tensorName, 2);
    }

    private static long[] parseLongArray(
            Object value, String field, String tensorName, int expectedSizeOrNegative) {
        List<?> list = requireList(value, field + " must be an array " + tensorContext(tensorName));
        if (expectedSizeOrNegative >= 0 && list.size() != expectedSizeOrNegative) {
            throw new SafetensorsFormatException(
                    field
                            + " must have exactly "
                            + expectedSizeOrNegative
                            + " values "
                            + tensorContext(tensorName));
        }
        long[] result = new long[list.size()];
        for (int i = 0; i < list.size(); i++) {
            result[i] = parseSizeT(list.get(i), field + "[" + i + "]", tensorName);
        }
        return result;
    }

    private static long parseSizeT(Object value, String field, String tensorName) {
        long parsed = parseIntegerNumber(value, field, tensorName);

        if (parsed < 0 || parsed > MAX_SIZE_T) {
            throw new SafetensorsFormatException(
                    field + " out of range " + tensorContext(tensorName));
        }
        return parsed;
    }

    private static long parseIntegerNumber(Object value, String field, String tensorName) {
        if (!(value instanceof Number)) {
            throw new SafetensorsFormatException(
                    field + " must be a number " + tensorContext(tensorName));
        }

        if (value instanceof BigDecimal) {
            try {
                return ((BigDecimal) value).toBigIntegerExact().longValueExact();
            } catch (ArithmeticException e) {
                throw new SafetensorsFormatException(
                        field + " must be an integer " + tensorContext(tensorName));
            }
        }
        if (value instanceof BigInteger) {
            BigInteger big = (BigInteger) value;
            if (big.compareTo(BigInteger.valueOf(Long.MIN_VALUE)) < 0
                    || big.compareTo(BigInteger.valueOf(Long.MAX_VALUE)) > 0) {
                throw new SafetensorsFormatException(
                        field + " out of range " + tensorContext(tensorName));
            }
            return big.longValue();
        }
        if (value instanceof Double || value instanceof Float) {
            double d = ((Number) value).doubleValue();
            if (!Double.isFinite(d)
                    || d != Math.rint(d)
                    || d < Long.MIN_VALUE
                    || d > Long.MAX_VALUE) {
                throw new SafetensorsFormatException(
                        field + " must be an integer " + tensorContext(tensorName));
            }
            return (long) d;
        }

        return ((Number) value).longValue();
    }

    private static Map<?, ?> requireObject(Object value, String message) {
        if (!(value instanceof Map)) {
            throw new SafetensorsFormatException(message);
        }
        return (Map<?, ?>) value;
    }

    private static List<?> requireList(Object value, String message) {
        if (!(value instanceof List)) {
            throw new SafetensorsFormatException(message);
        }
        return (List<?>) value;
    }

    private static String requireString(Object value, String message) {
        if (!(value instanceof String)) {
            throw new SafetensorsFormatException(message);
        }
        return (String) value;
    }

    private static String tensorContext(String tensorName) {
        return "for tensor '" + tensorName + "'";
    }

    private static void validateTensors(Collection<TensorEntry> tensors) {
        if (tensors.isEmpty()) {
            return;
        }

        // Sort by offset
        List<TensorEntry> sorted = new ArrayList<>(tensors);
        sorted.sort(Comparator.comparingLong(TensorEntry::byteOffset));

        // Check for overlaps
        long expectedOffset = 0;
        for (TensorEntry tensor : sorted) {
            if (tensor.byteOffset() < expectedOffset) {
                throw new SafetensorsFormatException("Overlapping tensors detected");
            }
            expectedOffset = tensor.byteOffset() + tensor.byteSize();
        }
    }

    private static int readFully(ReadableByteChannel channel, ByteBuffer buffer)
            throws IOException {
        int total = 0;
        while (buffer.hasRemaining()) {
            int read = channel.read(buffer);
            if (read < 0) {
                break;
            }
            total += read;
        }
        return total;
    }
}
