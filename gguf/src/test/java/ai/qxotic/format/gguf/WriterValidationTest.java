package ai.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Test;

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
    void testWriteFailsOnNonPositiveTensorDimension() {
        GGUF gguf =
                fakeGGUF(
                        3,
                        32,
                        0,
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        new LinkedHashMap<>(),
                        List.of(TensorEntry.create("t", new long[] {1, 0}, GGMLType.F32, 0)));

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
