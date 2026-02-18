package ai.qxotic.format.safetensors;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

public class SafetensorsBuilderTest extends SafetensorsTest {

    @Test
    public void testEmpty() {
        Safetensors st = Builder.newBuilder().build();
        assertEquals(32, st.getAlignment());
        assertTrue(st.getTensors().isEmpty());
        assertTrue(st.getMetadata().isEmpty());
        assertFalse(st.containsTensor("foo"));
    }

    @Test
    public void testMetadata() {
        Builder builder = Builder.newBuilder().putMetadataKey("foo", "bar");

        Safetensors st = builder.build();
        String fooValue = st.getMetadata().get("foo");
        assertEquals("bar", fooValue);
    }

    @Test
    public void testAlignment() {
        Builder builder = Builder.newBuilder().putMetadataKey("foo", "bar");
        assertEquals(32, builder.getAlignment());
        builder.setAlignment(4096);
        assertEquals(4096, builder.getAlignment());
        assertEquals("4096", builder.getMetadataValue("__alignment__"));

        assertTrue(builder.getMetadataKeys().contains("__alignment__"));
        builder.removeMetadataKey("__alignment__");
        assertEquals(32, builder.getAlignment());
    }

    @Test
    public void testFromSafetensors() {
        Safetensors expected = putMetadata(Builder.newBuilder()).build();
        Safetensors rebuilt = Builder.newBuilder(expected).build();
        assertEqualsSafetensors(expected, rebuilt, true);
    }

    @Test
    public void testFromSafetensorsRejectsNull() {
        assertThrows(NullPointerException.class, () -> Builder.newBuilder((Safetensors) null));
    }

    @Test
    public void testSetMetadataRejectsNulls() {
        Builder builder = Builder.newBuilder();
        assertThrows(NullPointerException.class, () -> builder.setMetadata(null));

        Map<String, String> withNullKey = new LinkedHashMap<>();
        withNullKey.put(null, "v");
        assertThrows(NullPointerException.class, () -> builder.setMetadata(withNullKey));

        Map<String, String> withNullValue = new LinkedHashMap<>();
        withNullValue.put("k", null);
        assertThrows(NullPointerException.class, () -> builder.setMetadata(withNullValue));
    }

    static Builder putMetadata(Builder builder) {
        return builder.putMetadataKey("format", "pt")
                .putMetadataKey("model_type", "llama")
                .putMetadataKey("version", "1.0");
    }

    @Test
    public void testMetadataKeysOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(key -> builder.putMetadataKey(key, key));
        assertEquals(keys, new ArrayList<>(builder.getMetadataKeys()));
    }

    @Test
    public void testTensorOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(
                key -> builder.putTensor(TensorEntry.create(key, DType.F32, new long[] {123}, 0)));
        assertEquals(
                keys,
                builder.getTensors().stream().map(TensorEntry::name).collect(Collectors.toList()));
    }

    @Test
    public void testReverseKeyOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(key -> builder.putMetadataKey(key, key));

        List<String> reversedKeys = new ArrayList<>(keys);
        Collections.reverse(reversedKeys);

        for (String key : reversedKeys) {
            String value = builder.getMetadataValue(key);
            builder.removeMetadataKey(key).putMetadataKey(key, value);
        }
        assertEquals(reversedKeys, new ArrayList<>(builder.getMetadataKeys()));
    }

    @Test
    public void testReverseTensorOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(
                key -> builder.putTensor(TensorEntry.create(key, DType.F32, new long[] {123}, 0)));

        List<TensorEntry> reversedTensors = new ArrayList<>(builder.getTensors());
        Collections.reverse(reversedTensors);

        for (TensorEntry tensor : reversedTensors) {
            builder.removeTensor(tensor.name()).putTensor(tensor);
        }
        assertEquals(reversedTensors, List.copyOf(builder.getTensors()));
    }

    @Test
    public void testPutRemoveTensors() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("foo", DType.F32, new long[] {1, 2}, 0));

        assertFalse(builder.containsTensor("absent"));

        assertTrue(builder.containsTensor("foo"));
        builder.removeTensor("foo");
        assertFalse(builder.containsTensor("foo"));
    }

    @Test
    public void testBuilderKeys() {
        Builder builder = Builder.newBuilder().putMetadataKey("foo", "bar");
        assertFalse(builder.getMetadataKeys().contains("absent"));
        assertTrue(builder.getMetadataKeys().contains("foo"));
        assertNotNull(builder.getMetadataValue("foo"));

        builder.removeMetadataKey("foo");

        assertFalse(builder.getMetadataKeys().contains("foo"));
        assertNull(builder.getMetadataValue("foo"));
    }

    @Test
    public void testBuilderClone() {
        Builder original =
                Builder.newBuilder()
                        .putMetadataKey("foo", "bar")
                        .putTensor(TensorEntry.create("tensor", DType.F32, new long[] {1}, 0))
                        .setAlignment(64);

        mutateAndCheck(original.clone(), original);
        mutateAndCheck(original, original.clone());
    }

    private static void mutateAndCheck(Builder toMutate, Builder toCheck) {
        toMutate.putMetadataKey("foo", "modified");
        toMutate.putTensor(TensorEntry.create("tensor", DType.F16, new long[] {1, 2, 3}, 0));
        toMutate.putTensor(TensorEntry.create("new_tensor", DType.I8, new long[] {3, 2, 1}, 0));
        toMutate.setAlignment(128);

        assertEquals("bar", toCheck.getMetadataValue("foo"));
        assertFalse(toCheck.containsTensor("new_tensor"));
        assertEquals(DType.F32, toCheck.getTensor("tensor").dtype());
        assertEquals(64, toCheck.getAlignment());
    }

    @Test
    public void testInvalidAlignment() {
        Builder builder = Builder.newBuilder();
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(0));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(-1));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(3));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(100));

        builder.putMetadataKey("__alignment__", "abc");
        assertThrows(IllegalArgumentException.class, builder::getAlignment);
    }

    @Test
    public void testValidAlignment() {
        Builder builder = Builder.newBuilder();
        builder.setAlignment(1);
        builder.setAlignment(2);
        builder.setAlignment(4);
        builder.setAlignment(8);
        builder.setAlignment(16);
        builder.setAlignment(32);
        builder.setAlignment(64);
        builder.setAlignment(128);
        builder.setAlignment(256);
        builder.setAlignment(512);
        builder.setAlignment(1024);
    }

    @Test
    public void testBuildWithoutRecomputingOffsets() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {10}, 100))
                        .putTensor(TensorEntry.create("tensor2", DType.F16, new long[] {5}, 500));

        Safetensors st = builder.build(false);

        assertEquals(100, st.getTensor("tensor1").byteOffset());
        assertEquals(500, st.getTensor("tensor2").byteOffset());
    }

    @Test
    public void testBuildWithRecomputingOffsets() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {10}, 999))
                        .putTensor(TensorEntry.create("tensor2", DType.F16, new long[] {5}, 888));

        Safetensors st = builder.build(true);

        assertEquals(0, st.getTensor("tensor1").byteOffset());
        assertEquals(64, st.getTensor("tensor2").byteOffset());
    }

    @Test
    public void testRecomputeOffsetsWithDefaultAlignment() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("small", DType.I8, new long[] {5}, 0))
                        .putTensor(TensorEntry.create("medium", DType.F16, new long[] {10}, 0))
                        .putTensor(TensorEntry.create("large", DType.F32, new long[] {20}, 0));

        Safetensors st = builder.build(true);
        assertEquals(32, st.getAlignment());

        TensorEntry small = st.getTensor("small");
        TensorEntry medium = st.getTensor("medium");
        TensorEntry large = st.getTensor("large");

        assertEquals(0, small.byteOffset());
        assertEquals(5, small.byteSize());

        assertEquals(32, medium.byteOffset());
        assertEquals(20, medium.byteSize());

        assertEquals(64, large.byteOffset());
        assertEquals(80, large.byteSize());
    }

    @Test
    public void testRecomputeOffsetsWithCustomAlignment() {
        Builder builder =
                Builder.newBuilder()
                        .setAlignment(64)
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {3}, 0))
                        .putTensor(TensorEntry.create("tensor2", DType.F32, new long[] {3}, 0));

        Safetensors st = builder.build(true);

        assertEquals(0, st.getTensor("tensor1").byteOffset());
        assertEquals(64, st.getTensor("tensor2").byteOffset());
    }

    @Test
    public void testRecomputeOffsetsWithAlignment1() {
        Builder builder =
                Builder.newBuilder()
                        .setAlignment(1)
                        .putTensor(TensorEntry.create("t1", DType.I8, new long[] {3}, 999))
                        .putTensor(TensorEntry.create("t2", DType.I8, new long[] {5}, 888))
                        .putTensor(TensorEntry.create("t3", DType.I8, new long[] {7}, 777));

        Safetensors st = builder.build(true);

        assertEquals(0, st.getTensor("t1").byteOffset());
        assertEquals(3, st.getTensor("t2").byteOffset());
        assertEquals(8, st.getTensor("t3").byteOffset());
    }

    @Test
    public void testDefaultBuildRecomputesOffsets() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("tensor1", DType.F32, new long[] {10}, 123))
                        .putTensor(TensorEntry.create("tensor2", DType.F16, new long[] {5}, 456));

        Safetensors st = builder.build();

        assertEquals(0, st.getTensor("tensor1").byteOffset());
        assertEquals(64, st.getTensor("tensor2").byteOffset());
    }
}
