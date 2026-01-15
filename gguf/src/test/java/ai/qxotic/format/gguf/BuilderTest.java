package ai.qxotic.format.gguf;

import org.junit.jupiter.api.Test;

import java.util.*;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

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
        keys.forEach(key -> builder.putTensor(TensorInfo.create(key, new long[]{123}, GGMLType.F32, 0)));
        assertEquals(keys, builder.getTensors().stream().map(TensorInfo::name).collect(Collectors.toList()));
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
        keys.forEach(key -> builder.putTensor(TensorInfo.create(key, new long[]{123}, GGMLType.F32, 0)));

        List<TensorInfo> reversedTensors = new ArrayList<>(builder.getTensors());
        Collections.reverse(reversedTensors);

        // Remove and insert the tensors in reverse order.
        for (TensorInfo tensor : reversedTensors) {
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
    public void testPutRemoveTensors() {
        Builder builder = Builder.newBuilder()
                .putTensor(TensorInfo.create("foo", new long[]{1, 2}, GGMLType.F32, -1));

        assertFalse(builder.containsTensor("absent"));

        assertTrue(builder.containsTensor("foo"));
        builder.removeTensor("foo");
        assertFalse(builder.containsTensor("foo"));
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
        Builder builder = putValues(Builder.newBuilder().putArrayOfString("array", new String[]{"foo"}));

        for (MetadataValueType valueType : MetadataValueType.values()) {
            assertEquals(valueType, builder.getType(valueType.name().toLowerCase()));
        }

        Set<String> expectedKeys = Arrays.stream(MetadataValueType.values())
                .map(MetadataValueType::name)
                .map(String::toLowerCase)
                .collect(Collectors.toSet());

        assertEquals(expectedKeys, builder.getMetadataKeys());
    }

    @Test
    public void testBuilderClone() {
        Builder original = Builder.newBuilder()
                .putString("foo", "bar")
                .putTensor(TensorInfo.create("tensor", new long[]{1}, GGMLType.F32, 0))
                .setVersion(2);

        // Changes in the copy do not affect the original.
        mutateAndCheck(original.clone(), original);

        // Changes in the original do not affect the copy.
        mutateAndCheck(original, original.clone());
    }

    private static void mutateAndCheck(Builder toMutate, Builder toCheck) {
        // Modify builder.
        toMutate.putInteger("foo", 123);
        toMutate.putTensor(TensorInfo.create("tensor", new long[]{1, 2, 3}, GGMLType.F16, 0)); // modify mutate
        toMutate.putTensor(TensorInfo.create("new_tensor", new long[]{3, 2, 1}, GGMLType.Q8_0, 0)); // modify mutate
        toMutate.setVersion(3);

        // Modifications to the mutated Builder do not affect copies.
        assertEquals(MetadataValueType.STRING, toCheck.getType("foo"));
        assertEquals("bar", toCheck.getValue(String.class, "foo"));
        assertFalse(toCheck.containsTensor("new_tensor"));
        assertEquals(GGMLType.F32, toCheck.getTensor("tensor").ggmlType());
        assertEquals(2, toCheck.getVersion());
    }

}
