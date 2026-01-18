package ai.qxotic.format.safetensors.impl;

import ai.qxotic.format.json.JSON;
import ai.qxotic.format.safetensors.DType;
import ai.qxotic.format.safetensors.Safetensors;
import ai.qxotic.format.safetensors.SafetensorsFormatException;
import ai.qxotic.format.safetensors.TensorEntry;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.*;

class ReaderImpl {
    static final String METADATA_KEY = "__metadata__";
    static final int HEADER_SIZE_BYTES = 8;
    static final int ALIGNMENT_DEFAULT_VALUE = 32;
    static final String ALIGNMENT_KEY = "__alignment__";

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
        Map<String, Object> header;
        try {
            header = (Map<String, Object>) JSON.parse(headerJson);
        } catch (JSON.ParseException e) {
            throw new SafetensorsFormatException("Invalid JSON in header: " + e.getMessage(), e);
        }

        // Extract metadata
        Map<String, String> metadata = new LinkedHashMap<>();
        Object metadataObj = header.get(METADATA_KEY);
        if (metadataObj != null) {
            if (!(metadataObj instanceof Map)) {
                throw new SafetensorsFormatException("__metadata__ must be an object");
            }
            Map<String, Object> metadataMap = (Map<String, Object>) metadataObj;
            for (Map.Entry<String, Object> entry : metadataMap.entrySet()) {
                if (!(entry.getValue() instanceof String)) {
                    throw new SafetensorsFormatException(
                            "__metadata__ values must be strings, got "
                                    + entry.getValue().getClass().getSimpleName()
                                    + " for key: "
                                    + entry.getKey());
                }
                metadata.put(entry.getKey(), (String) entry.getValue());
            }
        }

        // Extract tensors
        Map<String, TensorEntry> tensors = new LinkedHashMap<>();
        long dataOffset = HEADER_SIZE_BYTES + headerSize;

        for (Map.Entry<String, Object> entry : header.entrySet()) {
            String tensorName = entry.getKey();
            if (METADATA_KEY.equals(tensorName)) {
                continue;
            }
            TensorEntry tensorEntry = load(entry);
            tensors.put(tensorName, tensorEntry);
        }

        // Validate no overlaps in byte buffer
        validateTensors(tensors.values());

        return new SafetensorsImpl(dataOffset, metadata, tensors);
    }

    private static void validateTensorSchema(String tensorName, Map<String, Object> tensorEntry) {
        // Validate dtype field
        Object dtypeObj = tensorEntry.get("dtype");
        if (dtypeObj == null) {
            throw new SafetensorsFormatException("Missing dtype for tensor: " + tensorName);
        }
        if (!(dtypeObj instanceof String)) {
            throw new SafetensorsFormatException(
                    "dtype must be a string for tensor: "
                            + tensorName
                            + ", got "
                            + dtypeObj.getClass().getSimpleName());
        }

        // Validate shape field
        Object shapeObj = tensorEntry.get("shape");
        if (shapeObj == null) {
            throw new SafetensorsFormatException("Missing shape for tensor: " + tensorName);
        }
        if (!(shapeObj instanceof List)) {
            throw new SafetensorsFormatException(
                    "shape must be an array for tensor: "
                            + tensorName
                            + ", got "
                            + shapeObj.getClass().getSimpleName());
        }
        List<?> shapeList = (List<?>) shapeObj;
        for (int i = 0; i < shapeList.size(); i++) {
            if (!(shapeList.get(i) instanceof Number)) {
                throw new SafetensorsFormatException(
                        "shape[" + i + "] must be a number for tensor: " + tensorName);
            }
        }

        // Validate data_offsets field
        Object offsetsObj = tensorEntry.get("data_offsets");
        if (offsetsObj == null) {
            throw new SafetensorsFormatException("Missing data_offsets for tensor: " + tensorName);
        }
        if (!(offsetsObj instanceof List)) {
            throw new SafetensorsFormatException(
                    "data_offsets must be an array for tensor: "
                            + tensorName
                            + ", got "
                            + offsetsObj.getClass().getSimpleName());
        }
        List<?> offsetsList = (List<?>) offsetsObj;
        if (offsetsList.size() != 2) {
            throw new SafetensorsFormatException(
                    "data_offsets must have exactly 2 elements for tensor: "
                            + tensorName
                            + ", got "
                            + offsetsList.size());
        }
        if (!(offsetsList.get(0) instanceof Number)) {
            throw new SafetensorsFormatException(
                    "data_offsets[0] must be a number for tensor: " + tensorName);
        }
        if (!(offsetsList.get(1) instanceof Number)) {
            throw new SafetensorsFormatException(
                    "data_offsets[1] must be a number for tensor: " + tensorName);
        }
    }

    private static TensorEntry load(Map.Entry<String, Object> entry) {
        String tensorName = entry.getKey();
        if (!(entry.getValue() instanceof Map)) {
            throw new SafetensorsFormatException(
                    "Tensor entry must be a JSON object: " + tensorName);
        }
        Map<String, Object> tensorEntry = (Map<String, Object>) entry.getValue();

        // Validate schema first
        validateTensorSchema(tensorName, tensorEntry);

        // Parse dtype (already validated as String)
        String dtypeString = (String) tensorEntry.get("dtype");
        DType dtype;
        try {
            dtype = DType.valueOf(dtypeString);
        } catch (IllegalArgumentException e) {
            throw new SafetensorsFormatException(
                    "Unknown dtype '" + dtypeString + "' for tensor: " + tensorName);
        }

        // Parse shape (already validated as List<Number>)
        List<?> shapeList = (List<?>) tensorEntry.get("shape");
        long[] shape =
                shapeList.stream()
                        .map(n -> ((Number) n).longValue())
                        .mapToLong(Long::longValue)
                        .toArray();

        // Parse data_offsets (already validated as List<Number> with 2 elements)
        List<?> offsetsList = (List<?>) tensorEntry.get("data_offsets");
        long begin = ((Number) offsetsList.get(0)).longValue();
        long end = ((Number) offsetsList.get(1)).longValue();

        if (begin < 0 || end < 0) {
            throw new SafetensorsFormatException("Negative offsets for tensor: " + tensorName);
        }
        if (begin > end) {
            throw new SafetensorsFormatException(
                    "Invalid offsets: begin > end for tensor: " + tensorName);
        }

        long size = end - begin;
        long expectedSize = dtype.byteSizeForShape(shape);
        if (expectedSize != size) {
            throw new SafetensorsFormatException(
                    "Size mismatch for tensor "
                            + tensorName
                            + ": expected "
                            + expectedSize
                            + " bytes, got "
                            + size);
        }
        return TensorEntry.create(tensorName, dtype, shape, begin);
    }

    private static void validateTensors(Collection<TensorEntry> tensors) {
        if (tensors.isEmpty()) {
            return;
        }

        // Sort by offset
        List<TensorEntry> sorted = new ArrayList<>(tensors);
        sorted.sort(Comparator.comparingLong(TensorEntry::byteOffset));

        // Check for gaps and overlaps
        long expectedOffset = 0;
        for (TensorEntry tensor : sorted) {
            if (tensor.byteOffset() < expectedOffset) {
                throw new SafetensorsFormatException("Overlapping tensors detected");
            }
            //            if (tensor.byteOffset() > expectedOffset) {
            //                throw new SafetensorsFormatException("Gap in byte buffer at offset " +
            // expectedOffset);
            //            }
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
