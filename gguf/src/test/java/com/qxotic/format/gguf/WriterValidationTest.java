package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.io.ByteArrayOutputStream;
import java.nio.channels.Channels;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class WriterValidationTest extends GGUFTest {

    @Test
    void testWriteFailsOnLongTensorName() {
        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        0,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of(
                                TensorEntry.create(
                                        "a".repeat(65), new long[] {1}, GGMLType.F32, 0)));

        assertThrows(IllegalArgumentException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteTensorNameLimitIsGgmlMaxNameInUtf8Bytes() throws Exception {
        // ggml loads names of at most GGML_MAX_NAME - 1 = 63 bytes.
        writeToBytes(ggufWithTensorNamed("a".repeat(63)));
        assertThrows(
                IllegalArgumentException.class,
                () -> writeToBytes(ggufWithTensorNamed("a".repeat(64))));
        // 32 UTF-16 units, 64 UTF-8 bytes.
        assertThrows(
                IllegalArgumentException.class,
                () -> writeToBytes(ggufWithTensorNamed("é".repeat(32))));
    }

    private static GGUF ggufWithTensorNamed(String name) {
        return Builder.newBuilder()
                .putTensor(TensorEntry.create(name, new long[] {1}, GGMLType.F32, 0))
                .build();
    }

    @Test
    void testWriteFailsOnTooManyTensorDimensions() {
        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        0,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of(
                                TensorEntry.create(
                                        "t", new long[] {1, 2, 3, 4, 5}, GGMLType.F32, 0)));

        assertThrows(IllegalArgumentException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnMisalignedTensorOffset() {
        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        0,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of(TensorEntry.create("t", new long[] {1}, GGMLType.F32, 1)));

        assertThrows(IllegalArgumentException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnTooLongMetadataKey() {
        String key = "a".repeat(1 << 16);
        LinkedHashMap<String, Object> metadata = new LinkedHashMap<>();
        metadata.put(key, "v");
        LinkedHashMap<String, MetadataValueType> types = new LinkedHashMap<>();
        types.put(key, MetadataValueType.STRING);

        GGUF gguf = fakeGGUF(3, 32, 0, metadata, types, new LinkedHashMap<>(), List.of());
        assertThrows(IllegalArgumentException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnInvalidMetadataKeyFormat() {
        LinkedHashMap<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("Bad.Key", "v");
        LinkedHashMap<String, MetadataValueType> types = new LinkedHashMap<>();
        types.put("Bad.Key", MetadataValueType.STRING);

        GGUF gguf = fakeGGUF(3, 32, 0, metadata, types, new LinkedHashMap<>(), List.of());
        assertThrows(IllegalArgumentException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnMissingMetadataValue() {
        LinkedHashMap<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("k", null);
        LinkedHashMap<String, MetadataValueType> types = new LinkedHashMap<>();
        types.put("k", MetadataValueType.STRING);

        GGUF gguf = fakeGGUF(3, 32, 0, metadata, types, new LinkedHashMap<>(), List.of());
        assertThrows(IllegalStateException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnMissingMetadataType() {
        LinkedHashMap<String, Object> metadata = new LinkedHashMap<>();
        metadata.put("k", "v");

        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        0,
                        metadata,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of());
        assertThrows(IllegalStateException.class, () -> writeToBytes(gguf));
    }

    @Test
    void testWriteFailsOnTensorDataOffsetMismatch() {
        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        31,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of());

        assertThrows(IllegalStateException.class, () -> writeToBytes(gguf));
    }

    /** A GGUF the format refuses is refused whole: no byte reaches the channel, no file is made. */
    @Test
    void aRefusedWriteWritesNothing(@TempDir Path dir) {
        List<Builder> refused =
                List.of(
                        Builder.newBuilder()
                                .putTensor(
                                        TensorEntry.create(
                                                "a".repeat(64), new long[] {1}, GGMLType.F32, 0)),
                        Builder.newBuilder()
                                .putString("general.name", "x")
                                .putString("Bad.Key", "y"),
                        Builder.newBuilder()
                                .putTensor(
                                        TensorEntry.create(
                                                "t", new long[] {1, 1, 1, 1, 1}, GGMLType.F32, 0)));
        for (int i = 0; i < refused.size(); i++) {
            GGUF gguf = refused.get(i).build();
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GGUF.write(gguf, Channels.newChannel(out)));
            assertEquals(0, out.size(), "bytes sent for case " + i);
            Path file = dir.resolve(i + ".gguf");
            assertThrows(IllegalArgumentException.class, () -> GGUF.write(gguf, file));
            assertFalse(Files.exists(file), "a file left for case " + i);
        }
    }

    /** Version 3 is the only one built or written. */
    @Test
    void onlyVersionThreeIsWritten() {
        assertThrows(IllegalArgumentException.class, () -> Builder.newBuilder().setVersion(2));
        assertThrows(IllegalArgumentException.class, () -> Builder.newBuilder().setVersion(4));
        GGUF v2 =
                fakeGGUF(
                        2,
                        32,
                        32,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of());
        assertThrows(IllegalArgumentException.class, () -> writeToBytes(v2));
    }

    private static GGUF fakeGGUF(
            int version,
            int alignment,
            long tensorDataOffset,
            LinkedHashMap<String, Object> metadata,
            LinkedHashMap<String, MetadataValueType> metadataTypes,
            LinkedHashMap<String, MetadataValueType> componentTypes,
            List<TensorEntry> tensors) {
        return new GGUF() {
            private final LinkedHashMap<String, TensorEntry> tensorMap = toTensorMap(tensors);

            @Override
            public int getVersion() {
                return version;
            }

            @Override
            public int getAlignment() {
                return alignment;
            }

            @Override
            public long getTensorDataOffset() {
                return tensorDataOffset;
            }

            @Override
            public Set<String> getMetadataKeys() {
                return Collections.unmodifiableSet(metadata.keySet());
            }

            @SuppressWarnings("unchecked")
            @Override
            public <T> T getValue(Class<T> targetClass, String key) {
                return (T) metadata.get(key);
            }

            @Override
            public MetadataValueType getType(String key) {
                return metadataTypes.get(key);
            }

            @Override
            public MetadataValueType getComponentType(String key) {
                return componentTypes.get(key);
            }

            @Override
            public Collection<TensorEntry> getTensors() {
                return Collections.unmodifiableCollection(tensorMap.values());
            }

            @Override
            public TensorEntry getTensor(String tensorName) {
                return tensorMap.get(tensorName);
            }
        };
    }

    private static LinkedHashMap<String, TensorEntry> toTensorMap(List<TensorEntry> tensors) {
        LinkedHashMap<String, TensorEntry> map = new LinkedHashMap<>();
        for (TensorEntry tensor : tensors) {
            map.put(tensor.name(), tensor);
        }
        return map;
    }
}
