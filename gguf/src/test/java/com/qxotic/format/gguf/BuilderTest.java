package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;

public class BuilderTest extends GGUFTest {

    @Test
    public void testEmpty() {
        GGUF gguf = Builder.newBuilder().build();
        assertEquals(3, gguf.getVersion());
        assertEquals(32, gguf.getAlignment());
        assertTrue(gguf.getTensors().isEmpty());
        assertTrue(gguf.getMetadataKeys().isEmpty());
        assertFalse(gguf.containsKey("foo"));
        assertFalse(gguf.containsTensor("foo"));
    }

    @Test
    public void testString() {
        Builder builder = Builder.newBuilder().putString("foo", "bar");
        assertTrue(builder.containsKey("foo"));

        GGUF gguf = builder.build();
        assertEquals(MetadataValueType.STRING, gguf.getType("foo"));
        String fooValue = gguf.getValue(String.class, "foo");
        assertEquals("bar", fooValue);
    }

    @Test
    public void testAlignment() {
        Builder builder = Builder.newBuilder().putString("foo", "bar");
        assertFalse(builder.containsKey("general.alignment"));
        builder.setAlignment(4096);
        assertEquals(4096, builder.getAlignment());
        assertEquals(MetadataValueType.UINT32, builder.getType("general.alignment"));

        // Alignment is stored in the "general.alignment" key, removing the key restores the
        // alignment to its default value of 32.
        assertTrue(builder.containsKey("general.alignment"));
        builder.removeKey("general.alignment");
        assertEquals(32, builder.getAlignment());
    }

    @Test
    public void testAlignmentInvalidValues() {
        Builder builder = Builder.newBuilder();

        // Not power of 2
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(3));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(5));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(6));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(100));

        // Zero
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(0));

        // Negative
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(-1));
        assertThrows(IllegalArgumentException.class, () -> builder.setAlignment(-32));
    }

    @Test
    public void testGetAlignmentWithNonUint32Type() {
        Builder builder = Builder.newBuilder().putString("general.alignment", "bad");
        assertThrows(GGUFFormatException.class, () -> builder.getAlignment());
    }

    @Test
    public void testAlignmentPowerOfTwo() {
        Builder builder = Builder.newBuilder();

        // Valid powers of 2
        assertDoesNotThrow(() -> builder.setAlignment(1));
        assertDoesNotThrow(() -> builder.setAlignment(2));
        assertDoesNotThrow(() -> builder.setAlignment(4));
        assertDoesNotThrow(() -> builder.setAlignment(8));
        assertDoesNotThrow(() -> builder.setAlignment(16));
        assertDoesNotThrow(() -> builder.setAlignment(32));
        assertDoesNotThrow(() -> builder.setAlignment(64));
        assertDoesNotThrow(() -> builder.setAlignment(128));
        assertDoesNotThrow(() -> builder.setAlignment(256));
        assertDoesNotThrow(() -> builder.setAlignment(512));
        assertDoesNotThrow(() -> builder.setAlignment(1024));
        assertDoesNotThrow(() -> builder.setAlignment(2048));
        assertDoesNotThrow(() -> builder.setAlignment(4096));
    }

    @Test
    public void testFromGGUF() {
        GGUF expected = putValues(Builder.newBuilder()).build();
        GGUF rebuilt = Builder.newBuilder(expected).build();
        assertEqualsGGUF(expected, rebuilt, true);
    }

    @Test
    public void testNonExistingComponentType() {
        assertNull(Builder.newBuilder().getComponentType("does_not_exists"));
    }

    @Test
    public void testComponentTypeOrValues() {
        Builder builder = putValues(Builder.newBuilder());
        // Values (non-arrays) do not have a component type.
        for (String key : builder.getMetadataKeys()) {
            assertNotEquals(MetadataValueType.ARRAY, builder.getType(key));
            assertNull(builder.getComponentType(key));
        }
    }

    @Test
    public void testComponentTypeOfArrays() {
        Builder builder = putArrays(Builder.newBuilder());
        for (String key : builder.getMetadataKeys()) {
            assertEquals(MetadataValueType.ARRAY, builder.getType(key));
            MetadataValueType expectedComponentType = MetadataValueType.valueOf(key.toUpperCase());
            assertNotNull(expectedComponentType);
            assertEquals(expectedComponentType, builder.getComponentType(key));
        }
    }

    @Test
    public void testMetadataKeysOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(key -> builder.putString(key, key));
        assertEquals(keys, new ArrayList<>(builder.getMetadataKeys()));
    }

    @Test
    public void testTensorOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(
                key ->
                        builder.putTensor(
                                TensorEntry.create(key, new long[] {123}, GGMLType.F32, 0)));
        assertEquals(
                keys,
                builder.getTensors().stream().map(TensorEntry::name).collect(Collectors.toList()));
    }

    @Test
    public void testReverseKeyOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(key -> builder.putString(key, key));

        List<String> reversedKeys = new ArrayList<>(keys);
        Collections.reverse(reversedKeys);

        // Reinsert the keys in reverse order.
        for (String key : reversedKeys) {
            String value = builder.getValue(String.class, key);
            builder.removeKey(key).putString(key, value);
        }
        assertEquals(reversedKeys, new ArrayList<>(builder.getMetadataKeys()));
    }

    @Test
    public void testReverseTensorOrder() {
        List<String> keys = Arrays.asList("foo", "bar", "baz");
        Builder builder = Builder.newBuilder();
        keys.forEach(
                key ->
                        builder.putTensor(
                                TensorEntry.create(key, new long[] {123}, GGMLType.F32, 0)));

        List<TensorEntry> reversedTensors = new ArrayList<>(builder.getTensors());
        Collections.reverse(reversedTensors);

        // Remove and insert the tensors in reverse order.
        for (TensorEntry tensor : reversedTensors) {
            builder.removeTensor(tensor.name()).putTensor(tensor);
        }
        assertEquals(reversedTensors, List.copyOf(builder.getTensors()));
    }

    @Test
    public void testGetValue() {
        Builder builder = putValues(Builder.newBuilder());

        boolean z = builder.getValue(boolean.class, "bool");
        Boolean zboxed = builder.getValue(Boolean.class, "bool");

        byte b = builder.getValue(byte.class, "int8");
        Byte bboxed = builder.getValue(Byte.class, "int8");

        byte ub = builder.getValue(byte.class, "uint8");
        Byte ubboxed = builder.getValue(Byte.class, "uint8");

        short s = builder.getValue(short.class, "int16");
        Short sboxed = builder.getValue(Short.class, "int16");

        short us = builder.getValue(short.class, "uint16");
        Short usboxed = builder.getValue(Short.class, "uint16");

        int i = builder.getValue(int.class, "int32");
        Integer iboxed = builder.getValue(Integer.class, "int32");

        int ui = builder.getValue(int.class, "uint32");
        Integer uiboxed = builder.getValue(Integer.class, "uint32");

        long j = builder.getValue(long.class, "int64");
        Long jboxed = builder.getValue(Long.class, "int64");

        long js = builder.getValue(long.class, "uint64");
        Long jsboxed = builder.getValue(Long.class, "uint64");

        float f = builder.getValue(float.class, "float32");
        Float fboxed = builder.getValue(Float.class, "float32");

        double d = builder.getValue(double.class, "float64");
        Double dboxed = builder.getValue(Double.class, "float64");

        String string = builder.getValue(String.class, "string");

        assertThrows(ClassCastException.class, () -> builder.getValue(int.class, "bool"));
        assertThrows(ClassCastException.class, () -> builder.getValue(int.class, "bool"));
    }

    interface GetValue {
        <T> T apply(Class<T> targetClass, String key);
    }

    void testGetValueContractValues(GetValue getValue) {
        assertNull(getValue.apply(boolean.class, "absent"));
        assertNull(getValue.apply(byte.class, "absent"));
        assertNull(getValue.apply(short.class, "absent"));
        assertNull(getValue.apply(int.class, "absent"));
        assertNull(getValue.apply(long.class, "absent"));
        assertNull(getValue.apply(float.class, "absent"));
        assertNull(getValue.apply(double.class, "absent"));
        assertNull(getValue.apply(String.class, "absent"));

        assertNull(getValue.apply(Boolean.class, "absent"));
        assertNull(getValue.apply(Byte.class, "absent"));
        assertNull(getValue.apply(Short.class, "absent"));
        assertNull(getValue.apply(Integer.class, "absent"));
        assertNull(getValue.apply(Long.class, "absent"));
        assertNull(getValue.apply(Float.class, "absent"));
        assertNull(getValue.apply(Double.class, "absent"));

        assertNull(getValue.apply(boolean[].class, "absent"));
        assertNull(getValue.apply(byte[].class, "absent"));
        assertNull(getValue.apply(short[].class, "absent"));
        assertNull(getValue.apply(int[].class, "absent"));
        assertNull(getValue.apply(long[].class, "absent"));
        assertNull(getValue.apply(float[].class, "absent"));
        assertNull(getValue.apply(double[].class, "absent"));
        assertNull(getValue.apply(String[].class, "absent"));

        boolean z = getValue.apply(boolean.class, "bool");
        Boolean zboxed = getValue.apply(Boolean.class, "bool");
        assertNotNull(zboxed);

        byte b = getValue.apply(byte.class, "int8");
        Byte bboxed = getValue.apply(Byte.class, "int8");
        assertNotNull(bboxed);

        byte ub = getValue.apply(byte.class, "uint8");
        Byte ubboxed = getValue.apply(Byte.class, "uint8");
        assertNotNull(ubboxed);

        short s = getValue.apply(short.class, "int16");
        Short sboxed = getValue.apply(Short.class, "int16");
        assertNotNull(sboxed);

        short us = getValue.apply(short.class, "uint16");
        Short usboxed = getValue.apply(Short.class, "uint16");
        assertNotNull(usboxed);

        int i = getValue.apply(int.class, "int32");
        Integer iboxed = getValue.apply(Integer.class, "int32");
        assertNotNull(iboxed);

        int ui = getValue.apply(int.class, "uint32");
        Integer uiboxed = getValue.apply(Integer.class, "uint32");
        assertNotNull(uiboxed);

        long j = getValue.apply(long.class, "int64");
        Long jboxed = getValue.apply(Long.class, "int64");
        assertNotNull(jboxed);

        long uj = getValue.apply(long.class, "uint64");
        Long ujboxed = getValue.apply(Long.class, "uint64");
        assertNotNull(ujboxed);

        float f = getValue.apply(float.class, "float32");
        Float fboxed = getValue.apply(Float.class, "float32");
        assertNotNull(fboxed);

        double d = getValue.apply(double.class, "float64");
        Double dboxed = getValue.apply(Double.class, "float64");
        assertNotNull(dboxed);

        String string = getValue.apply(String.class, "string");
        assertNotNull(string);

        assertThrows(ClassCastException.class, () -> getValue.apply(int.class, "bool"));
        assertThrows(ClassCastException.class, () -> getValue.apply(String.class, "bool"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Integer.class, "string"));

        assertThrows(ClassCastException.class, () -> getValue.apply(boolean[].class, "bool"));
        assertThrows(ClassCastException.class, () -> getValue.apply(byte[].class, "int8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(byte[].class, "uint8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(short[].class, "int16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(short[].class, "uint16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(int[].class, "int32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(int[].class, "uint32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(long[].class, "int64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(long[].class, "uint64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(float[].class, "float32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(double[].class, "float64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(String[].class, "string"));

        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "string"));
        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "int8"));
        assertNotNull(getValue.apply(Object.class, "uint8"));
        assertNotNull(getValue.apply(Object.class, "int16"));
        assertNotNull(getValue.apply(Object.class, "uint16"));
        assertNotNull(getValue.apply(Object.class, "int32"));
        assertNotNull(getValue.apply(Object.class, "uint32"));
        assertNotNull(getValue.apply(Object.class, "int64"));
        assertNotNull(getValue.apply(Object.class, "uint64"));
        assertNotNull(getValue.apply(Object.class, "float32"));
        assertNotNull(getValue.apply(Object.class, "float64"));
        assertNotNull(getValue.apply(Object.class, "string"));
    }

    void testGetValueContractArrays(GetValue getValue) {
        assertThrows(ClassCastException.class, () -> getValue.apply(boolean.class, "bool"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Boolean.class, "bool"));
        assertThrows(ClassCastException.class, () -> getValue.apply(byte.class, "int8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Byte.class, "int8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(byte.class, "uint8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Byte.class, "uint8"));
        assertThrows(ClassCastException.class, () -> getValue.apply(short.class, "int16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Short.class, "int16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(short.class, "uint16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Short.class, "uint16"));
        assertThrows(ClassCastException.class, () -> getValue.apply(int.class, "int32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Integer.class, "int32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(int.class, "uint32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Integer.class, "int32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(long.class, "int64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Long.class, "int64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(long.class, "uint64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Long.class, "uint64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(float.class, "float32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Float.class, "float32"));
        assertThrows(ClassCastException.class, () -> getValue.apply(double.class, "float64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(Double.class, "float64"));
        assertThrows(ClassCastException.class, () -> getValue.apply(String.class, "string"));

        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "string"));
        assertNotNull(getValue.apply(Object.class, "bool"));
        assertNotNull(getValue.apply(Object.class, "int8"));
        assertNotNull(getValue.apply(Object.class, "uint8"));
        assertNotNull(getValue.apply(Object.class, "int16"));
        assertNotNull(getValue.apply(Object.class, "uint16"));
        assertNotNull(getValue.apply(Object.class, "int32"));
        assertNotNull(getValue.apply(Object.class, "uint32"));
        assertNotNull(getValue.apply(Object.class, "int64"));
        assertNotNull(getValue.apply(Object.class, "uint64"));
        assertNotNull(getValue.apply(Object.class, "float32"));
        assertNotNull(getValue.apply(Object.class, "float64"));
        assertNotNull(getValue.apply(Object.class, "string"));
    }

    @Test
    public void testGetValueBuilder() {
        testGetValueContractValues(putValues(Builder.newBuilder())::getValue);
        testGetValueContractArrays(putArrays(Builder.newBuilder())::getValue);
    }

    @Test
    public void testGetValueGGUF() {
        testGetValueContractValues(putValues(Builder.newBuilder()).build()::getValue);
        testGetValueContractArrays(putArrays(Builder.newBuilder()).build()::getValue);
    }

    @Test
    public void testGetStringShortcut() throws IOException {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "my-model")
                        .putString("general.description", "A test model")
                        .putInteger("llama.context_length", 4096)
                        .build();

        // Test getString returns correct value
        assertEquals("my-model", gguf.getString("general.name"));
        assertEquals("A test model", gguf.getString("general.description"));

        // Test getString returns null for missing key
        assertNull(gguf.getString("missing.key"));

        // Test getString throws ClassCastException for non-string value
        assertThrows(ClassCastException.class, () -> gguf.getString("llama.context_length"));

        // Test round-trip
        GGUF read = readFromBytes(writeToBytes(gguf));
        assertEquals("my-model", read.getString("general.name"));
        assertNull(read.getString("missing.key"));
    }

    @Test
    public void testGetStringOrDefaultShortcut() throws IOException {
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "my-model")
                        .putInteger("llama.context_length", 4096)
                        .build();

        // Test getString with default returns value when key exists
        assertEquals("my-model", gguf.getStringOrDefault("general.name", "default"));

        // Test getString with default returns default when key missing
        assertEquals("default", gguf.getStringOrDefault("missing.key", "default"));
        assertEquals("", gguf.getStringOrDefault("missing.key", ""));

        // Test getString with default throws ClassCastException for non-string
        assertThrows(
                ClassCastException.class,
                () -> gguf.getStringOrDefault("llama.context_length", "default"));

        // Test round-trip
        GGUF read = readFromBytes(writeToBytes(gguf));
        assertEquals("my-model", read.getStringOrDefault("general.name", "default"));
        assertEquals("default", read.getStringOrDefault("missing.key", "default"));
    }

    @Test
    public void testPutRemoveTensors() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("foo", new long[] {1, 2}, GGMLType.F32, -1));

        assertFalse(builder.containsTensor("absent"));

        assertTrue(builder.containsTensor("foo"));
        builder.removeTensor("foo");
        assertFalse(builder.containsTensor("foo"));
    }

    @Test
    public void testTensorAbsoluteOffset() throws IOException {
        // Create a GGUF with tensor data offset
        GGUF gguf =
                Builder.newBuilder()
                        .putString("general.name", "test")
                        .putTensor(TensorEntry.create("tensor1", new long[] {1}, GGMLType.F32, 0))
                        .putTensor(TensorEntry.create("tensor2", new long[] {1}, GGMLType.F32, 128))
                        .build();

        TensorEntry tensor1 = gguf.getTensor("tensor1");
        TensorEntry tensor2 = gguf.getTensor("tensor2");

        // absoluteOffset = tensorDataOffset + tensor.offset()
        long expectedAbsolute1 = gguf.getTensorDataOffset() + tensor1.offset();
        long expectedAbsolute2 = gguf.getTensorDataOffset() + tensor2.offset();

        assertEquals(expectedAbsolute1, gguf.absoluteOffset(tensor1));
        assertEquals(expectedAbsolute2, gguf.absoluteOffset(tensor2));

        // Verify absolute offset is correct by checking offset difference
        assertEquals(
                tensor2.offset() - tensor1.offset(),
                gguf.absoluteOffset(tensor2) - gguf.absoluteOffset(tensor1));
    }

    @Test
    public void testTensorAbsoluteOffsetRoundTrip() throws IOException {
        // Build and write GGUF
        GGUF original =
                Builder.newBuilder()
                        .putString("key", "value")
                        .putTensor(TensorEntry.create("weights", new long[] {100}, GGMLType.F32, 0))
                        .build();

        // Write and read back
        GGUF read = readFromBytes(writeToBytes(original));

        TensorEntry tensor = read.getTensor("weights");

        // absoluteOffset should work correctly on deserialized GGUF
        long expectedAbsolute = read.getTensorDataOffset() + tensor.offset();
        assertEquals(expectedAbsolute, read.absoluteOffset(tensor));
    }

    @Test
    public void testBuilderKeys() {
        Builder builder = Builder.newBuilder().putString("foo", "bar");
        assertFalse(builder.containsKey("absent"));
        assertTrue(builder.containsKey("foo"));
        assertNotNull(builder.getValue(Object.class, "foo"));

        builder.removeKey("foo");

        assertFalse(builder.containsKey("foo"));
        assertNull(builder.getValue(Object.class, "foo"));
    }

    @Test
    public void testBuilderTypes() {
        Builder builder =
                putValues(Builder.newBuilder().putArrayOfString("array", new String[] {"foo"}));

        for (MetadataValueType valueType : MetadataValueType.values()) {
            assertEquals(valueType, builder.getType(valueType.name().toLowerCase()));
        }

        Set<String> expectedKeys =
                Arrays.stream(MetadataValueType.values())
                        .map(MetadataValueType::name)
                        .map(String::toLowerCase)
                        .collect(Collectors.toSet());

        assertEquals(expectedKeys, builder.getMetadataKeys());
    }

    @Test
    public void testTypeTransitionScalarToArraySameKey() {
        Builder builder = Builder.newBuilder().putString("k", "v");
        assertEquals(MetadataValueType.STRING, builder.getType("k"));
        assertNull(builder.getComponentType("k"));

        builder.putArrayOfInteger("k", new int[] {1, 2, 3});
        assertEquals(MetadataValueType.ARRAY, builder.getType("k"));
        assertEquals(MetadataValueType.INT32, builder.getComponentType("k"));
        assertArrayEquals(new int[] {1, 2, 3}, builder.getValue(int[].class, "k"));
    }

    @Test
    public void testTypeTransitionArrayToScalarSameKey() {
        Builder builder = Builder.newBuilder().putArrayOfInteger("k", new int[] {1, 2, 3});
        assertEquals(MetadataValueType.ARRAY, builder.getType("k"));
        assertEquals(MetadataValueType.INT32, builder.getComponentType("k"));

        builder.putString("k", "v");
        assertEquals(MetadataValueType.STRING, builder.getType("k"));
        assertNull(builder.getComponentType("k"));
        assertEquals("v", builder.getValue(String.class, "k"));
    }

    @Test
    public void testRemoveReinsertDifferentKindRoundTrip() throws IOException {
        Builder builder = Builder.newBuilder().putArrayOfInteger("k", new int[] {1, 2});
        builder.removeKey("k");
        builder.putString("k", "v");

        GGUF gguf = readFromBytes(writeToBytes(builder.build()));
        assertEquals(MetadataValueType.STRING, gguf.getType("k"));
        assertNull(gguf.getComponentType("k"));
        assertEquals("v", gguf.getValue(String.class, "k"));

        Builder builder2 = Builder.newBuilder().putString("k", "v");
        builder2.removeKey("k");
        builder2.putArrayOfLong("k", new long[] {7L, 8L});

        GGUF gguf2 = readFromBytes(writeToBytes(builder2.build()));
        assertEquals(MetadataValueType.ARRAY, gguf2.getType("k"));
        assertEquals(MetadataValueType.INT64, gguf2.getComponentType("k"));
        assertArrayEquals(new long[] {7L, 8L}, gguf2.getValue(long[].class, "k"));
    }

    @Test
    public void testEmptyArraysAllTypesRoundTrip() throws IOException {
        GGUF gguf =
                Builder.newBuilder()
                        .putArrayOfString("string", new String[0])
                        .putArrayOfBoolean("bool", new boolean[0])
                        .putArrayOfByte("int8", new byte[0])
                        .putArrayOfUnsignedByte("uint8", new byte[0])
                        .putArrayOfShort("int16", new short[0])
                        .putArrayOfUnsignedShort("uint16", new short[0])
                        .putArrayOfInteger("int32", new int[0])
                        .putArrayOfUnsignedInteger("uint32", new int[0])
                        .putArrayOfLong("int64", new long[0])
                        .putArrayOfUnsignedLong("uint64", new long[0])
                        .putArrayOfFloat("float32", new float[0])
                        .putArrayOfDouble("float64", new double[0])
                        .build();

        GGUF read = readFromBytes(writeToBytes(gguf));
        for (String key : gguf.getMetadataKeys()) {
            assertEquals(MetadataValueType.ARRAY, read.getType(key));
            assertNotNull(read.getComponentType(key));
            assertEquals(0, Array.getLength(read.getValue(Object.class, key)));
        }
    }

    @Test
    public void testArrayOfStringWithNullElementFailsOnWrite() {
        assertThrows(
                NullPointerException.class,
                () ->
                        Builder.newBuilder()
                                .putArrayOfString("s", new String[] {"ok", null})
                                .build());
    }

    @Test
    public void testBuilderClone() {
        Builder original =
                Builder.newBuilder()
                        .putString("foo", "bar")
                        .putTensor(TensorEntry.create("tensor", new long[] {1}, GGMLType.F32, 0))
                        .setVersion(2);

        // Changes in the copy do not affect the original.
        mutateAndCheck(original.clone(), original);

        // Changes in the original do not affect the copy.
        mutateAndCheck(original, original.clone());
    }

    private static void mutateAndCheck(Builder toMutate, Builder toCheck) {
        // Modify builder.
        toMutate.putInteger("foo", 123);
        toMutate.putTensor(
                TensorEntry.create(
                        "tensor", new long[] {1, 2, 3}, GGMLType.F16, 0)); // modify mutate
        toMutate.putTensor(
                TensorEntry.create(
                        "new_tensor", new long[] {3, 2, 1}, GGMLType.Q8_0, 0)); // modify mutate
        toMutate.setVersion(3);

        // Modifications to the mutated Builder do not affect copies.
        assertEquals(MetadataValueType.STRING, toCheck.getType("foo"));
        assertEquals("bar", toCheck.getValue(String.class, "foo"));
        assertFalse(toCheck.containsTensor("new_tensor"));
        assertEquals(GGMLType.F32, toCheck.getTensor("tensor").ggmlType());
        assertEquals(2, toCheck.getVersion());
    }

    @Test
    public void testBuildWithoutRecomputingOffsets() {
        TensorEntry t1 = TensorEntry.create("t1", new long[] {10}, GGMLType.F32, 100);
        TensorEntry t2 = TensorEntry.create("t2", new long[] {20}, GGMLType.F32, 200);

        GGUF gguf = Builder.newBuilder().putTensor(t1).putTensor(t2).build(false);

        // Original offsets preserved
        assertEquals(100, gguf.getTensor("t1").offset());
        assertEquals(200, gguf.getTensor("t2").offset());
    }

    @Test
    public void testBuildWithRecomputingOffsets() {
        TensorEntry t1 = TensorEntry.create("t1", new long[] {10}, GGMLType.F32, 100);
        TensorEntry t2 = TensorEntry.create("t2", new long[] {20}, GGMLType.F32, 200);

        GGUF gguf = Builder.newBuilder().putTensor(t1).putTensor(t2).build(true);

        // Offsets recomputed (start at 0 with alignment)
        assertEquals(0, gguf.getTensor("t1").offset());
        // t2 starts after t1's data (40 bytes) aligned to 32 = 64
        assertEquals(64, gguf.getTensor("t2").offset());
    }

    @Test
    public void testBuildDefaultRecomputesOffsets() {
        TensorEntry t1 = TensorEntry.create("t1", new long[] {10}, GGMLType.F32, 100);

        GGUF gguf = Builder.newBuilder().putTensor(t1).build();

        // Default build() should recompute offsets
        assertEquals(0, gguf.getTensor("t1").offset());
    }

    @Test
    public void testRemoveNonExistentKey() {
        Builder builder = Builder.newBuilder().putString("key", "value");

        // Should not throw
        assertDoesNotThrow(() -> builder.removeKey("nonexistent"));

        // Original key still there
        assertTrue(builder.containsKey("key"));
    }

    @Test
    public void testRemoveNonExistentTensor() {
        Builder builder =
                Builder.newBuilder()
                        .putTensor(TensorEntry.create("t", new long[] {1}, GGMLType.F32, 0));

        // Should not throw
        assertDoesNotThrow(() -> builder.removeTensor("nonexistent"));

        // Original tensor still there
        assertTrue(builder.containsTensor("t"));
    }
}
