package com.qxotic.jota;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class JotaShapeTest {

    @Test
    void testScalar() {
        Shape scalar = Shape.of();
        assertTrue(scalar.isScalar());
        assertTrue(scalar.isFlat());
        assertEquals(0, scalar.rank());
        assertEquals(0, scalar.flatRank());
        assertArrayEquals(new long[0], scalar.toArray());
        assertEquals("[]", scalar.toString());
        assertEquals(scalar, scalar.flatten());
    }

    @Test
    void testSingleton() {
        Shape shape = Shape.of(42);
        assertFalse(shape.isScalar());
        assertTrue(shape.isFlat());
        assertEquals(1, shape.rank());
        assertEquals(1, shape.flatRank());
        assertArrayEquals(new long[]{42}, shape.toArray());
        assertEquals("[42]", shape.toString());
        assertEquals(shape, shape.flatten());
    }
}
