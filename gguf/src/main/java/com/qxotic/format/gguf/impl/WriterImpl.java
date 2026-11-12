package com.qxotic.format.gguf.impl;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.MetadataValueType;
import com.qxotic.format.gguf.TensorEntry;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.lang.reflect.Array;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.WritableByteChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;

/**
 * Writes a GGUF header in memory first: one the format refuses fails before the channel or the file
 * gets its first byte, and the channel gets one write instead of one per value.
 */
final class WriterImpl {
    private final ByteBuffer scratch =
            ByteBuffer.allocate(Long.BYTES).order(ByteOrder.LITTLE_ENDIAN);
    private final ByteArrayOutputStream header = new ByteArrayOutputStream();
    private final GGUF gguf;

    private WriterImpl(GGUF gguf) {
        this.gguf = gguf;
    }

    static void write(GGUF gguf, WritableByteChannel byteChannel) throws IOException {
        ByteBuffer bytes = ByteBuffer.wrap(serialize(gguf));
        while (bytes.hasRemaining()) {
            byteChannel.write(bytes);
        }
    }

    static void write(GGUF gguf, Path path) throws IOException {
        Files.write(path, serialize(gguf), StandardOpenOption.CREATE_NEW);
    }

    /** The header, up to {@link GGUF#getTensorDataOffset()}; refused as a whole or not at all. */
    private static byte[] serialize(GGUF gguf) {
        WriterImpl writer = new WriterImpl(gguf);
        writer.writeHeader();
        for (TensorEntry tensorEntry : gguf.getTensors()) {
            writer.writeTensorEntry(tensorEntry);
        }
        // padding up to the tensor data, even when there are no tensors
        writer.header.writeBytes(
                new byte[(int) GGUFImpl.padding(writer.header.size(), gguf.getAlignment())]);
        if (writer.header.size() != gguf.getTensorDataOffset()) {
            throw new IllegalStateException(
                    "tensorDataOffset mismatch: the header takes "
                            + writer.header.size()
                            + " bytes, the GGUF says "
                            + gguf.getTensorDataOffset());
        }
        return writer.header.toByteArray();
    }

    private void writeHeader() {
        if (gguf.getVersion() != ReaderImpl.GGUF_VERSION) {
            throw new IllegalArgumentException(
                    "GGUF version "
                            + gguf.getVersion()
                            + " cannot be written: only version "
                            + ReaderImpl.GGUF_VERSION);
        }
        writeInt(ReaderImpl.GGUF_MAGIC); // uint32_t magic: "GGUF" little-endian
        writeInt(gguf.getVersion()); // uint32_t version
        writeLong(gguf.getTensors().size()); // uint64_t tensor_count
        writeLong(gguf.getMetadataKeys().size()); // uint64_t metadata_kv_count
        for (String key : gguf.getMetadataKeys()) {
            if (key.length() >= (1 << 16)) {
                throw new IllegalArgumentException("Metadata key too long: " + key.length());
            }
            if (!ReaderImpl.isValidKey(key)) {
                throw new IllegalArgumentException("Invalid metadata key format: " + key);
            }
            Object value = gguf.getValue(Object.class, key);
            if (value == null) {
                throw new IllegalStateException("Missing metadata value for key: " + key);
            }
            MetadataValueType valueType = gguf.getType(key);
            if (valueType == null) {
                throw new IllegalStateException("Missing metadata type for key: " + key);
            }
            writeString(key);
            if (valueType == MetadataValueType.ARRAY) {
                writeArray(gguf.getComponentType(key), value);
            } else {
                writeValueType(valueType);
                writeValue(valueType, value);
            }
        }
    }

    private void writeTensorEntry(TensorEntry tensorEntry) {
        String name = tensorEntry.name();
        byte[] nameBytes = name.getBytes(StandardCharsets.UTF_8);
        if (nameBytes.length > ReaderImpl.MAX_TENSOR_NAME_BYTES) {
            throw new IllegalArgumentException(
                    "Tensor name too long (>"
                            + ReaderImpl.MAX_TENSOR_NAME_BYTES
                            + " bytes): "
                            + nameBytes.length
                            + " for tensor '"
                            + name
                            + "'");
        }
        long[] shape = tensorEntry.shape();
        if (shape.length > 4) {
            throw new IllegalArgumentException(
                    "Tensor has too many dimensions: "
                            + shape.length
                            + " for tensor '"
                            + name
                            + "'");
        }
        if (tensorEntry.offset() % gguf.getAlignment() != 0) {
            throw new IllegalArgumentException(
                    "Tensor offset "
                            + tensorEntry.offset()
                            + " is not aligned to "
                            + gguf.getAlignment()
                            + " for tensor '"
                            + name
                            + "'");
        }
        writeLong(nameBytes.length); // gguf_string_t name
        header.writeBytes(nameBytes);
        writeInt(shape.length); // uint32_t n_dimensions
        for (long dimension : shape) {
            writeLong(dimension); // uint64_t shape[n_dimensions]
        }
        writeInt(tensorEntry.ggmlType().getId()); // ggml_type type
        writeLong(tensorEntry.offset()); // uint64_t offset, relative to the tensor data
    }

    private void writeArray(MetadataValueType componentType, Object value) {
        writeValueType(MetadataValueType.ARRAY);
        writeValueType(componentType);
        writeLong(Array.getLength(value)); // number of elements, not bytes
        switch (componentType) {
            case UINT8: // fall-through
            case INT8:
                header.writeBytes((byte[]) value);
                break;
            case UINT16: // fall-through
            case INT16:
                for (short s : (short[]) value) {
                    writeShort(s);
                }
                break;
            case UINT32: // fall-through
            case INT32:
                for (int i : (int[]) value) {
                    writeInt(i);
                }
                break;
            case UINT64: // fall-through
            case INT64:
                for (long n : (long[]) value) {
                    writeLong(n);
                }
                break;
            case FLOAT32:
                for (float f : (float[]) value) {
                    writeInt(Float.floatToRawIntBits(f));
                }
                break;
            case FLOAT64:
                for (double d : (double[]) value) {
                    writeLong(Double.doubleToRawLongBits(d));
                }
                break;
            case BOOL:
                for (boolean b : (boolean[]) value) {
                    header.write(b ? 1 : 0);
                }
                break;
            case STRING:
                for (String s : (String[]) value) {
                    writeString(s);
                }
                break;
            case ARRAY:
                throw new UnsupportedOperationException("array of arrays");
        }
    }

    private void writeValue(MetadataValueType valueType, Object value) {
        switch (valueType) {
            case UINT8: // fall-through
            case INT8:
                header.write((byte) value);
                break;
            case UINT16: // fall-through
            case INT16:
                writeShort((short) value);
                break;
            case UINT32: // fall-through
            case INT32:
                writeInt((int) value);
                break;
            case UINT64: // fall-through
            case INT64:
                writeLong((long) value);
                break;
            case FLOAT32:
                writeInt(Float.floatToRawIntBits((float) value));
                break;
            case FLOAT64:
                writeLong(Double.doubleToRawLongBits((double) value));
                break;
            case BOOL:
                header.write((boolean) value ? 1 : 0);
                break;
            case STRING:
                writeString((String) value);
                break;
            case ARRAY:
                throw new IllegalArgumentException("an array is written by writeArray");
        }
    }

    private void writeString(String string) {
        byte[] bytes = string.getBytes(StandardCharsets.UTF_8);
        writeLong(bytes.length); // uint64_t len, in bytes
        header.writeBytes(bytes); // UTF-8, not NUL-terminated
    }

    @SuppressWarnings("EnumOrdinal")
    private void writeValueType(MetadataValueType valueType) {
        writeInt(valueType.ordinal());
    }

    private void writeShort(short value) {
        header.write(scratch.clear().putShort(value).array(), 0, Short.BYTES);
    }

    private void writeInt(int value) {
        header.write(scratch.clear().putInt(value).array(), 0, Integer.BYTES);
    }

    private void writeLong(long value) {
        header.write(scratch.clear().putLong(value).array(), 0, Long.BYTES);
    }
}
