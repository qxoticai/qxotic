package ai.qxotic.format.gguf.impl;

import ai.qxotic.format.gguf.GGMLType;
import ai.qxotic.format.gguf.GGUF;
import ai.qxotic.format.gguf.MetadataValueType;
import ai.qxotic.format.gguf.TensorInfo;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.ReadableByteChannel;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

final class ReaderImpl {
    static final int GGUF_MAGIC = 0x46554747;
    static final int ALIGNMENT_DEFAULT_VALUE = 32; // must be a power of 2
    static final String ALIGNMENT_KEY = "general.alignment";

    private static final List<Integer> SUPPORTED_GGUF_VERSIONS = List.of(2, 3);
    private int version;
    private int alignment;
    private Map<String, Object> metadata;
    private Map<String, MetadataValueType> metadataTypes;
    private long totalBytesRead;

    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.nativeOrder());

    GGUF readImpl(ReadableByteChannel byteChannel) throws IOException {
        // The header of the file.
        int tensorCount = readHeader(byteChannel); // gguf_header_t header;
        // Tensor infos, which can be used to locate the tensor data.
        // gguf_tensor_info_t tensor_infos[header.tensor_count];
        Map<String, TensorInfo> tensorInfos = new LinkedHashMap<>(tensorCount);
        for (int i = 0; i < tensorCount; ++i) {
            TensorInfo ti = readTensorInfo(byteChannel);
            assert !tensorInfos.containsKey(ti.name());
            tensorInfos.put(ti.name(), ti);
        }
        // Padding to the nearest multiple of `ALIGNMENT`.
        // uint8_t _padding[ALIGNMENT - (sizeof(header + tensor_infos) % ALIGNMENT)];
        // long _padding = -byteChannel.position() & (ALIGNMENT - 1);
        int padding = (int) GGUFImpl.padding(totalBytesRead, getAlignment());

        // Tensor data.
        //
        // This is arbitrary binary data corresponding to the weights of the model. This data should be
        // close
        // or identical to the data in the original model file, but may be different due to quantization
        // or
        // other optimizations for inference. Any such deviations should be recorded in the metadata or
        // as
        // part of the architecture definition.
        //
        // Each tensor's data must be stored within this array, and located through its `tensor_infos`
        // entry.
        // The offset of each tensor's data must be a multiple of `ALIGNMENT`, and the space between
        // tensors
        // should be padded to `ALIGNMENT` bytes.
        // uint8_t tensor_data[];
        long tensorDataOffset = totalBytesRead + padding;
        return new GGUFImpl(this.version, tensorDataOffset, this.metadata, this.metadataTypes, tensorInfos);
    }

    private GGMLType readGGMLType(ReadableByteChannel byteChannel) throws IOException {
        int ggmlTypeId = readInt(byteChannel); // ggml_type type;
        return GGMLType.fromId(ggmlTypeId);
    }

    private TensorInfo readTensorInfo(ReadableByteChannel byteChannel) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = readString(byteChannel); // gguf_string_t name;
        assert name.length() <= 64;
        // The number of shape in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = readInt(byteChannel); // uint32_t n_dimensions;
        assert n_dimensions <= 4;
        // The shape of the tensor.
        long[] dimensions = new long[n_dimensions]; // uint64_t shape[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            dimensions[i] = readLong(byteChannel);
        }
        // The type of the tensor.
        GGMLType ggmlType = readGGMLType(byteChannel); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = readLong(byteChannel); // uint64_t offset;
        assert offset % getAlignment() == 0;
        return TensorInfo.create(name, dimensions, ggmlType, offset);
    }

    private String readString(ReadableByteChannel byteChannel) throws IOException {
        // A string in GGUF.
        // The length of the string, in bytes.
        int len = Math.toIntExact(readLong(byteChannel)); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        byte[] bytes = new byte[len]; // char string[len];
        readBytes(byteChannel, bytes);
        return new String(bytes, StandardCharsets.UTF_8);
    }

    private int readHeader(ReadableByteChannel byteChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        int magic = readInt(byteChannel); //    uint32_t magic;
        if (magic != GGUF_MAGIC) {
            throw new IllegalArgumentException("Invalid header.magic: " + magic + " expected: " + GGUF_MAGIC);
        }
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        this.version = readInt(byteChannel); // uint32_t version;
        if (!SUPPORTED_GGUF_VERSIONS.contains(version)) {
            throw new IllegalArgumentException(
                    "Unsupported header.version:" + version + " expected: " + SUPPORTED_GGUF_VERSIONS);
        }
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        int tensorCount = Math.toIntExact(readLong(byteChannel)); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        int metadataKeyValueCount = Math.toIntExact(readLong(byteChannel)); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        this.metadata = new LinkedHashMap<>(metadataKeyValueCount);
        this.metadataTypes = new LinkedHashMap<>(metadataKeyValueCount);
        for (int i = 0; i < metadataKeyValueCount; ++i) {
            // The key of the metadata. It is a standard GGUF string, with the following caveats:
            // - It must be a valid ASCII string.
            // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by
            // a `.`.
            // - It must be at most 2^16-1/65535 bytes long.
            // Any keys that do not follow these rules are invalid.
            String key = readString(byteChannel); // gguf_string_t key;
            assert key.length() < (1 << 16);
            assert key.codePoints()
                    .allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');

            // The type of the value.
            // Must be one of the `gguf_metadata_value_type` values.
            MetadataValueType valueType = readMetadataValueType(byteChannel); // gguf_metadata_value_type value_type;
            // The value.
            Object value = readMetadataValueOfType(byteChannel, key, valueType);
            assert !metadata.containsKey(key);
            assert !metadataTypes.containsKey(key);

            metadata.put(key, value);
            metadataTypes.put(key, valueType);
        }

        return tensorCount;
    }

    private Object readArray(ReadableByteChannel byteChannel, String key) throws IOException {
        // Any value type is valid, including arrays.
        MetadataValueType componentType = readMetadataValueType(byteChannel); // gguf_metadata_value_type type;
        // Record the component type.
        this.metadataTypes.put(key + "[]", componentType);
        // Number of elements, not bytes.
        int len = Math.toIntExact(readLong(byteChannel)); // uint64_t len;
        // The array of values.
        // gguf_metadata_value_t array[len];
        switch (componentType) {
            case UINT8:
            case INT8:
                byte[] bytes = new byte[len];
                for (int i = 0; i < len; ++i) {
                    bytes[i] = readByte(byteChannel);
                }
                return bytes;
            case UINT16:
            case INT16:
                short[] shorts = new short[len];
                for (int i = 0; i < len; ++i) {
                    shorts[i] = readShort(byteChannel);
                }
                return shorts;
            case UINT32:
            case INT32:
                int[] ints = new int[len];
                for (int i = 0; i < len; ++i) {
                    ints[i] = readInt(byteChannel);
                }
                return ints;
            case UINT64:
            case INT64:
                long[] longs = new long[len];
                for (int i = 0; i < len; ++i) {
                    longs[i] = readLong(byteChannel);
                }
                return longs;
            case FLOAT32:
                float[] floats = new float[len];
                for (int i = 0; i < len; ++i) {
                    floats[i] = readFloat(byteChannel);
                }
                return floats;
            case FLOAT64:
                double[] doubles = new double[len];
                for (int i = 0; i < len; ++i) {
                    doubles[i] = readDouble(byteChannel);
                }
                return doubles;
            case BOOL:
                boolean[] booleans = new boolean[len];
                for (int i = 0; i < len; ++i) {
                    booleans[i] = readBoolean(byteChannel);
                }
                return booleans;
            case STRING:
                String[] strings = new String[len];
                for (int i = 0; i < len; ++i) {
                    strings[i] = readString(byteChannel);
                }
                return strings;
            case ARRAY:
                throw new UnsupportedOperationException("Cannot read array of arrays");
            default:
                throw new UnsupportedOperationException("Found array of unknown type " + componentType);
        }
    }

    private Object readMetadataValueOfType(ReadableByteChannel byteChannel, String key, MetadataValueType valueType)
            throws IOException {
        switch (valueType) {
            case UINT8: // fall-through
            case INT8:
                return readByte(byteChannel);
            case UINT16: // fall-through
            case INT16:
                return readShort(byteChannel);
            case UINT32: // fall-through
            case INT32:
                return readInt(byteChannel);
            case FLOAT32:
                return readFloat(byteChannel);
            case UINT64: // fall-through
            case INT64:
                return readLong(byteChannel);
            case FLOAT64:
                return readDouble(byteChannel);
            case BOOL:
                return readBoolean(byteChannel);
            case STRING:
                return readString(byteChannel);
            case ARRAY:
                return readArray(byteChannel, key);
            default:
                throw new IllegalArgumentException();
        }
    }

    private ByteBuffer readFully(ReadableByteChannel byteChannel, ByteBuffer byteBuffer) throws IOException {
        while (byteBuffer.position() < byteBuffer.limit()) {
            int bytesRead = byteChannel.read(byteBuffer);
            if (bytesRead < 0) {
                throw new EOFException();
            } else if (bytesRead > 0) {
                totalBytesRead += bytesRead;
            }
        }
        return byteBuffer;
    }

    private void readBytes(ReadableByteChannel byteChannel, byte[] bytes) throws IOException {
        readFully(byteChannel, ByteBuffer.wrap(bytes));
    }

    private byte readByte(ReadableByteChannel byteChannel) throws IOException {
        return readFully(byteChannel, BB_8.clear().limit(1)).get(0);
    }

    private boolean readBoolean(ReadableByteChannel byteChannel) throws IOException {
        return readByte(byteChannel) != 0;
    }

    private short readShort(ReadableByteChannel byteChannel) throws IOException {
        return readFully(byteChannel, BB_8.clear().limit(2)).getShort(0);
    }

    private int readInt(ReadableByteChannel byteChannel) throws IOException {
        return readFully(byteChannel, BB_8.clear().limit(4)).getInt(0);
    }

    private long readLong(ReadableByteChannel byteChannel) throws IOException {
        return readFully(byteChannel, BB_8.clear().limit(8)).getLong(0);
    }

    private float readFloat(ReadableByteChannel byteChannel) throws IOException {
        return Float.intBitsToFloat(readInt(byteChannel));
    }

    private double readDouble(ReadableByteChannel byteChannel) throws IOException {
        return Double.longBitsToDouble(readLong(byteChannel));
    }

    private MetadataValueType readMetadataValueType(ReadableByteChannel byteChannel) throws IOException {
        int index = readInt(byteChannel);
        return MetadataValueType.fromIndex(index);
    }

    private int getAlignment() {
        if (alignment != 0) {
            return alignment;
        }
        assert !metadataTypes.containsKey(ALIGNMENT_KEY)
                || metadataTypes.get(ALIGNMENT_KEY) == MetadataValueType.UINT32;
        alignment = (int) metadata.getOrDefault(ALIGNMENT_KEY, ALIGNMENT_DEFAULT_VALUE);
        assert Integer.bitCount(alignment) == 1 : "alignment must be a power of two";
        assert alignment > 0;
        return alignment;
    }
}
