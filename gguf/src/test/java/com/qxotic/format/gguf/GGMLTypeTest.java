package com.qxotic.format.gguf;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

public class GGMLTypeTest {

    @Test
    public void testGetBitsPerWeight() {
        assertEquals(32.0, GGMLType.F32.getBitsPerWeight(), 0.001);
        assertEquals(16.0, GGMLType.F16.getBitsPerWeight(), 0.001);
        assertEquals(16.0, GGMLType.BF16.getBitsPerWeight(), 0.001);
        assertEquals(8.0, GGMLType.I8.getBitsPerWeight(), 0.001);
        assertEquals(4.5, GGMLType.Q4_0.getBitsPerWeight(), 0.001);
        assertEquals(5.0, GGMLType.Q4_1.getBitsPerWeight(), 0.001);
        assertEquals(5.5, GGMLType.Q5_0.getBitsPerWeight(), 0.001);
        assertEquals(6.0, GGMLType.Q5_1.getBitsPerWeight(), 0.001);
        assertEquals(8.5, GGMLType.Q8_0.getBitsPerWeight(), 0.001);
        assertEquals(9.0, GGMLType.Q8_1.getBitsPerWeight(), 0.001);
        assertEquals(2.625, GGMLType.Q2_K.getBitsPerWeight(), 0.001);
        assertEquals(3.4375, GGMLType.Q3_K.getBitsPerWeight(), 0.001);
        assertEquals(4.5, GGMLType.Q4_K.getBitsPerWeight(), 0.001);
        assertEquals(5.5, GGMLType.Q5_K.getBitsPerWeight(), 0.001);
        assertEquals(6.5625, GGMLType.Q6_K.getBitsPerWeight(), 0.001);
        assertEquals(9.125, GGMLType.Q8_K.getBitsPerWeight(), 0.001);
        assertEquals(2.0625, GGMLType.IQ2_XXS.getBitsPerWeight(), 0.001);
        assertEquals(2.3125, GGMLType.IQ2_XS.getBitsPerWeight(), 0.001);
    }

    @Test
    public void testIsQuantized() {
        assertFalse(GGMLType.F32.isQuantized());
        assertFalse(GGMLType.F16.isQuantized());
        assertFalse(GGMLType.BF16.isQuantized());
        assertFalse(GGMLType.I8.isQuantized());
        assertFalse(GGMLType.I16.isQuantized());
        assertFalse(GGMLType.I32.isQuantized());
        assertFalse(GGMLType.I64.isQuantized());
        assertFalse(GGMLType.F64.isQuantized());

        assertTrue(GGMLType.Q4_0.isQuantized());
        assertTrue(GGMLType.Q4_1.isQuantized());
        assertTrue(GGMLType.Q5_0.isQuantized());
        assertTrue(GGMLType.Q5_1.isQuantized());
        assertTrue(GGMLType.Q8_0.isQuantized());
        assertTrue(GGMLType.Q8_1.isQuantized());
        assertTrue(GGMLType.Q2_K.isQuantized());
        assertTrue(GGMLType.Q3_K.isQuantized());
        assertTrue(GGMLType.Q4_K.isQuantized());
        assertTrue(GGMLType.Q5_K.isQuantized());
        assertTrue(GGMLType.Q6_K.isQuantized());
        assertTrue(GGMLType.Q8_K.isQuantized());
    }

    @Test
    public void testByteSizeForValid() {
        assertEquals(4, GGMLType.F32.byteSizeFor(1));
        assertEquals(40, GGMLType.F32.byteSizeFor(10));
        assertEquals(18, GGMLType.Q4_0.byteSizeFor(32));
        assertEquals(36, GGMLType.Q4_0.byteSizeFor(64));
        assertEquals(84, GGMLType.Q2_K.byteSizeFor(256));
        assertEquals(168, GGMLType.Q2_K.byteSizeFor(512));
    }

    @Test
    public void testByteSizeForInvalidElementCount() {
        assertThrows(
                IllegalArgumentException.class, () -> GGMLType.Q4_0.byteSizeFor(33), "33 elements");
        assertThrows(
                IllegalArgumentException.class, () -> GGMLType.Q4_0.byteSizeFor(1), "1 element");
        assertThrows(
                IllegalArgumentException.class, () -> GGMLType.Q4_0.byteSizeFor(31), "31 elements");
        assertThrows(
                IllegalArgumentException.class,
                () -> GGMLType.Q2_K.byteSizeFor(100),
                "100 elements");
        assertThrows(
                IllegalArgumentException.class,
                () -> GGMLType.Q2_K.byteSizeFor(255),
                "255 elements");
    }

    @Test
    public void testByteSizeForOverflow() {
        assertThrows(ArithmeticException.class, () -> GGMLType.F32.byteSizeFor(Long.MAX_VALUE));
    }

    @Test
    public void testElementsForByteSize() {
        assertEquals(1, GGMLType.F32.elementsForByteSize(4));
        assertEquals(10, GGMLType.F32.elementsForByteSize(40));
        assertEquals(32, GGMLType.Q4_0.elementsForByteSize(18));
        assertEquals(64, GGMLType.Q4_0.elementsForByteSize(36));
        assertEquals(256, GGMLType.Q2_K.elementsForByteSize(84));
    }

    @Test
    public void testElementsForByteSizeInvalid() {
        assertThrows(IllegalArgumentException.class, () -> GGMLType.F32.elementsForByteSize(5));
        assertThrows(IllegalArgumentException.class, () -> GGMLType.Q4_0.elementsForByteSize(17));
        assertThrows(IllegalArgumentException.class, () -> GGMLType.Q2_K.elementsForByteSize(100));
    }

    @Test
    public void testFromId() {
        assertEquals(GGMLType.F32, GGMLType.fromId(0));
        assertEquals(GGMLType.F16, GGMLType.fromId(1));
        assertEquals(GGMLType.Q4_0, GGMLType.fromId(2));

        assertThrows(ArrayIndexOutOfBoundsException.class, () -> GGMLType.fromId(-1));
        assertThrows(ArrayIndexOutOfBoundsException.class, () -> GGMLType.fromId(1000));
    }

    @Test
    public void testGetId() {
        assertEquals(0, GGMLType.F32.getId());
        assertEquals(1, GGMLType.F16.getId());
        assertEquals(2, GGMLType.Q4_0.getId());
    }

    @Test
    public void testGetBlockByteSize() {
        assertEquals(4, GGMLType.F32.getBlockByteSize());
        assertEquals(2, GGMLType.F16.getBlockByteSize());
        assertEquals(18, GGMLType.Q4_0.getBlockByteSize());
        assertEquals(84, GGMLType.Q2_K.getBlockByteSize());
    }

    @Test
    public void testGetElementsPerBlock() {
        assertEquals(1, GGMLType.F32.getElementsPerBlock());
        assertEquals(1, GGMLType.F16.getElementsPerBlock());
        assertEquals(32, GGMLType.Q4_0.getElementsPerBlock());
        assertEquals(256, GGMLType.Q2_K.getElementsPerBlock());
    }

    @Test
    public void testDeprecatedTypesAreMarked() throws NoSuchFieldException {
        // Check that deprecated enum constants have @Deprecated annotation
        assertTrue(
                GGMLType.class.getField("Q4_2").isAnnotationPresent(Deprecated.class),
                "Q4_2 should be deprecated");
        assertTrue(
                GGMLType.class.getField("Q4_3").isAnnotationPresent(Deprecated.class),
                "Q4_3 should be deprecated");
        assertTrue(
                GGMLType.class.getField("Q4_0_4_4").isAnnotationPresent(Deprecated.class),
                "Q4_0_4_4 should be deprecated");
        assertTrue(
                GGMLType.class.getField("Q4_0_4_8").isAnnotationPresent(Deprecated.class),
                "Q4_0_4_8 should be deprecated");
        assertTrue(
                GGMLType.class.getField("Q4_0_8_8").isAnnotationPresent(Deprecated.class),
                "Q4_0_8_8 should be deprecated");
        assertTrue(
                GGMLType.class.getField("IQ4_NL_4_4").isAnnotationPresent(Deprecated.class),
                "IQ4_NL_4_4 should be deprecated");
        assertTrue(
                GGMLType.class.getField("IQ4_NL_4_8").isAnnotationPresent(Deprecated.class),
                "IQ4_NL_4_8 should be deprecated");
        assertTrue(
                GGMLType.class.getField("IQ4_NL_8_8").isAnnotationPresent(Deprecated.class),
                "IQ4_NL_8_8 should be deprecated");
    }
}
