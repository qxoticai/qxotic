package com.llm4j.gguf.impl;

import com.llm4j.gguf.GGMLType;
import com.llm4j.gguf.GGUF;
import com.llm4j.gguf.MetadataValueType;
import com.llm4j.gguf.TensorInfo;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;

final class WriterImpl {
    private final ByteBuffer BB_8 = ByteBuffer.allocate(Long.BYTES).order(ByteOrder.nativeOrder());

    private final GGUF gguf;
    private long totalBytesWritten;

    private WriterImpl(GGUF gguf) {
        this.gguf = gguf;
    }

    static void writeImpl(GGUF gguf, WritableByteChannel byteChannel) throws IOException {
        WriterImpl writer = new WriterImpl(gguf);
        writer.writeHeader(byteChannel);
        for (TensorInfo tensorInfo : gguf.getTensors()) {
            writer.writeTensorInfo(byteChannel, tensorInfo);
        }
        // Always align, even if there are no tensors.
        writer.writePaddingForAlignment(byteChannel);

        assert writer.totalBytesWritten == gguf.getTensorDataOffset();
    }

    private void writeFully(WritableByteChannel byteChannel, ByteBuffer byteBuffer) throws IOException {
        while (byteBuffer.hasRemaining()) {
            this.totalBytesWritten += byteChannel.write(byteBuffer);
        }
    }

    private void writePaddingForAlignment(WritableByteChannel byteChannel) throws IOException {
        int padding = (int) GGUFImpl.padding(this.totalBytesWritten, gguf.getAlignment());
        writeFully(byteChannel, ByteBuffer.allocate(padding));
    }

    private void writeLong(WritableByteChannel byteChannel, long value) throws IOException {
        writeFully(byteChannel, BB_8.clear().putLong(value).flip());
    }

    private void writeDouble(WritableByteChannel byteChannel, double value) throws IOException {
        writeLong(byteChannel, Double.doubleToRawLongBits(value));
    }

    private void writeInt(WritableByteChannel byteChannel, int value) throws IOException {
        writeFully(byteChannel, BB_8.clear().putInt(value).flip());
    }

    private void writeFloat(WritableByteChannel byteChannel, float value) throws IOException {
        writeInt(byteChannel, Float.floatToRawIntBits(value));
    }

    private void writeByte(WritableByteChannel byteChannel, byte value) throws IOException {
        writeFully(byteChannel, BB_8.clear().put(value).flip());
    }

    private void writeBoolean(WritableByteChannel byteChannel, boolean value) throws IOException {
        writeByte(byteChannel, value ? (byte) 1 : (byte) 0);
    }

    private void writeShort(WritableByteChannel byteChannel, short value) throws IOException {
        writeFully(byteChannel, BB_8.clear().putShort(value).flip());
    }

    private void writeBytes(WritableByteChannel byteChannel, byte[] bytes) throws IOException {
        writeFully(byteChannel, ByteBuffer.wrap(bytes));
    }

    private void writeString(WritableByteChannel byteChannel, String string) throws IOException {
        byte[] bytes = string.getBytes(StandardCharsets.UTF_8);
        // A string in GGUF.
        // The length of the string, in bytes.
        writeLong(byteChannel, bytes.length); // uint64_t len;
        // The string as a UTF-8 non-null-terminated string.
        writeBytes(byteChannel, bytes);
    }

    private void writeTensorInfo(WritableByteChannel byteChannel, TensorInfo tensorInfo) throws IOException {
        // The name of the tensor. It is a standard GGUF string, with the caveat that
        // it must be at most 64 bytes long.
        String name = tensorInfo.name();
        assert name.length() <= 64;
        writeString(byteChannel, name); // gguf_string_t name;
        // The number of shape in the tensor.
        // Currently at most 4, but this may change in the future.
        int n_dimensions = tensorInfo.shape().length;
        assert n_dimensions <= 4;
        writeInt(byteChannel, n_dimensions); // uint32_t n_dimensions;
        // The shape of the tensor.
        long[] dimensions = tensorInfo.shape(); // uint64_t shape[n_dimensions];
        for (int i = 0; i < n_dimensions; ++i) {
            assert dimensions[i] > 0;
            writeLong(byteChannel, dimensions[i]);
        }
        // The type of the tensor.
        GGMLType ggmlType = tensorInfo.ggmlType();
        writeGGMLType(byteChannel, ggmlType); // ggml_type type;
        // The offset of the tensor's data in this file in bytes.
        // This offset is relative to `tensor_data`, not to the start
        // of the file, to make it easier for writers to write the file.
        // Readers should consider exposing this offset relative to the
        // file to make it easier to read the data.
        // Must be a multiple of `ALIGNMENT`.
        long offset = tensorInfo.offset();
        writeLong(byteChannel, offset); // uint64_t offset;
        assert offset % gguf.getAlignment() == 0;
        // return new GGUFTensorInfo(name, dimensions, ggmlType, offset);
    }

    @SuppressWarnings("EnumOrdinal")
    private void writeGGMLType(WritableByteChannel byteChannel, GGMLType ggmlType) throws IOException {
        writeInt(byteChannel, ggmlType.ordinal());
    }

    private void writeHeader(WritableByteChannel byteChannel) throws IOException {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        writeInt(byteChannel, ReaderImpl.GGUF_MAGIC); //    uint32_t magic;
        // The version of the format implemented.
        // Must be `3` for version described in this spec.
        //
        // This version should only be increased for structural changes to the format.
        // Changes that do not affect the structure of the file should instead update the metadata
        // to signify the change.
        writeInt(byteChannel, gguf.getVersion()); // uint32_t version;
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        writeLong(byteChannel, gguf.getTensors().size()); // uint64_t tensor_count;
        // The number of metadata key-value pairs.
        writeLong(byteChannel, gguf.getMetadataKeys().size()); // uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
        for (String key : gguf.getMetadataKeys()) {
            // The key of the metadata. It is a standard GGUF string, with the following caveats:
            // - It must be a valid ASCII string.
            // - It must be a hierarchical key, where each segment is `lower_snake_case` and separated by
            // a `.`.
            // - It must be at most 2^16-1/65535 bytes long.
            // Any keys that do not follow these rules are invalid.
            assert key.length() < (1 << 16);
            assert key.codePoints()
                    .allMatch(cp -> ('a' <= cp && cp <= 'z') || ('0' <= cp && cp <= '9') || cp == '_' || cp == '.');
            writeString(byteChannel, key);
            Object value = gguf.getValue(Object.class, key);
            assert value != null;

            MetadataValueType valueType = gguf.getType(key);
            assert valueType != null;

            if (valueType == MetadataValueType.ARRAY) {
                MetadataValueType componentType = gguf.getComponentType(key);
                writeTypedArrayOf(byteChannel, componentType, value);
            } else {
                writeTypedValue(byteChannel, valueType, value);
            }
        }
    }

    private void writeTypedArrayOf(WritableByteChannel byteChannel, MetadataValueType componentType, Object value)
            throws IOException {
        int arrayLength = Array.getLength(value);
        writeValueType(byteChannel, MetadataValueType.ARRAY);
        writeValueType(byteChannel, componentType);
        writeLong(byteChannel, arrayLength);
        switch (componentType) {
            case UINT8: // fall-through
            case INT8:
                writeBytes(byteChannel, (byte[]) value);
                break;
            case UINT16: // fall-through
            case INT16:
                for (short s : (short[]) value) {
                    writeShort(byteChannel, s);
                }
                break;
            case UINT32: // fall-through
            case INT32:
                for (int i : (int[]) value) {
                    writeInt(byteChannel, i);
                }
                break;
            case BOOL:
                for (boolean b : (boolean[]) value) {
                    writeBoolean(byteChannel, b);
                }
                break;
            case STRING:
                String[] stringArray = (String[]) value;
                for (String s : stringArray) {
                    writeString(byteChannel, s);
                }
                break;
            case UINT64: // fall-through
            case INT64:
                for (long n : (long[]) value) {
                    writeLong(byteChannel, n);
                }
                break;
            case FLOAT32:
                for (float f : (float[]) value) {
                    writeFloat(byteChannel, f);
                }
                break;
            case FLOAT64:
                for (double d : (double[]) value) {
                    writeDouble(byteChannel, d);
                }
                break;
            case ARRAY:
                throw new UnsupportedOperationException("array of arrays");
        }
    }

    private void writeTypedValue(WritableByteChannel byteChannel, MetadataValueType valueType, Object value)
            throws IOException {
        if (valueType == MetadataValueType.ARRAY) {
            throw new IllegalArgumentException("use writeArrayOf instead");
        }
        writeValueType(byteChannel, valueType);
        switch (valueType) {
            case UINT8: // fall-through
            case INT8:
                writeByte(byteChannel, (byte) value);
                break;
            case UINT16: // fall-through
            case INT16:
                writeShort(byteChannel, (short) value);
                break;
            case UINT32: // fall-through
            case INT32:
                writeInt(byteChannel, (int) value);
                break;
            case FLOAT32:
                writeFloat(byteChannel, (float) value);
                break;
            case BOOL:
                writeBoolean(byteChannel, (boolean) value);
                break;
            case STRING:
                writeString(byteChannel, (String) value);
                break;
            case ARRAY:
                throw new IllegalArgumentException("use writeArrayOf instead");
            case UINT64: // fall-through
            case INT64:
                writeLong(byteChannel, (long) value);
                break;
            case FLOAT64:
                writeDouble(byteChannel, (double) value);
                break;
        }
    }

    @SuppressWarnings("EnumOrdinal")
    private void writeValueType(WritableByteChannel byteChannel, MetadataValueType valueType) throws IOException {
        writeInt(byteChannel, valueType.ordinal());
    }
}
