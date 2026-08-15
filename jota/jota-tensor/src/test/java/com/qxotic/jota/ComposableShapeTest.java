package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class ComposableShapeTest {

    @Test
    void testFlatShapeWithInts() {
        // Should use of(int...) overload
        Shape shape = Shape.of(1, 2, 3);

        assertTrue(shape.isFlat());
        assertEquals(3, shape.rank());
        assertArrayEquals(new long[] {1, 2, 3}, shape.toArray());
    }

    @Test
    void testFlatShapeWithLongs() {
        // Should use of(long...) overload
        Shape shape = Shape.of(1, 2L, 3L);

        assertTrue(shape.isFlat());
        assertEquals(3, shape.rank());
        assertArrayEquals(new long[] {1, 2, 3}, shape.toArray());
    }

    @Test
    void testSimpleNested() {
        // Shape.of(2, Shape.of(4, 5), 6) → (2, (4, 5), 6)
        Shape shape = Shape.of(2, Shape.of(4, 5), 6);

        assertEquals(3, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[] {2, 4, 5, 6}, shape.toArray());

        // Check nesting structure
        assertEquals(2, shape.size(0));
        assertEquals(4 * 5, shape.size(1));
        assertEquals(6, shape.size(2));

        System.out.println("Simple nested: " + shape);
    }

    @Test
    void testSingleElementShapesNormalized() {
        // Shape.of(1, Shape.of(2), Shape.of(3)) → (1, 2, 3)
        // Single-element shapes get unwrapped
        Shape shape = Shape.of(1, Shape.of(2), Shape.of(3));

        assertTrue(shape.isFlat());
        assertEquals(3, shape.rank());
        assertArrayEquals(new long[] {1, 2, 3}, shape.toArray());

        System.out.println("Normalized flat: " + shape);
    }

    @Test
    void testSingleShapeArgumentUnwrapped() {
        // Shape.of(Shape.of(2, 3)) → (2, 3)
        Shape inner = Shape.of(2, 3);
        Shape shape = Shape.of(inner);

        assertSame(inner, shape); // Should return the same instance
        assertTrue(shape.isFlat());
        assertEquals(2, shape.rank());
        assertArrayEquals(new long[] {2, 3}, shape.toArray());
    }

    @Test
    void testSingleShapeArgumentUnwrappedNested() {
        // Shape.of(Shape.of(1, Shape.of(2, 3))) → (1, (2, 3))
        Shape inner = Shape.of(1, Shape.of(2, 3));
        Shape shape = Shape.of(inner);

        assertSame(inner, shape);
        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
    }

    @Test
    void testMixedComposition() {
        // Shape.of(Shape.of(2), Shape.of(3, 4)) → (2, (3, 4))
        Shape shape = Shape.of(Shape.of(2), Shape.of(3, 4));

        assertEquals(2, shape.rank());
        assertEquals(3, shape.flatRank());
        assertArrayEquals(new long[] {2, 3, 4}, shape.toArray());

        assertEquals(2, shape.size(0));
        assertEquals(3 * 4, shape.size(1));
    }

    @Test
    void testMultipleNestedGroups() {
        // Shape.of(Shape.of(1, 2), Shape.of(3, 4)) → ((1, 2), (3, 4))
        Shape shape = Shape.of(Shape.of(1, 2), Shape.of(3, 4));

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[] {1, 2, 3, 4}, shape.toArray());

        assertEquals(1 * 2, shape.size(0));
        assertEquals(3 * 4, shape.size(1));

        System.out.println("Multiple nested: " + shape);
    }

    @Test
    void testDeepNesting() {
        // Shape.of(1, Shape.of(2, Shape.of(3, 4))) → (1, (2, (3, 4)))
        Shape shape = Shape.of(1, Shape.of(2, Shape.of(3, 4)));

        assertEquals(2, shape.rank());
        assertEquals(4, shape.flatRank());
        assertArrayEquals(new long[] {1, 2, 3, 4}, shape.toArray());

        System.out.println("Deep nesting: " + shape);

        // Check mode structure
        Shape mode1 = shape.modeAt(1);
        assertEquals(2, mode1.rank());
        assertEquals(3, mode1.flatRank());
        assertArrayEquals(new long[] {2, 3, 4}, mode1.toArray());
    }

    @Test
    void testComplexComposition() {
        // (1, 2, (3, 4, 5))
        Shape shape = Shape.of(1, 2, Shape.of(3, 4, 5));

        assertEquals(3, shape.rank());
        assertEquals(5, shape.flatRank());
        assertArrayEquals(new long[] {1, 2, 3, 4, 5}, shape.toArray());

        assertEquals(1, shape.size(0));
        assertEquals(2, shape.size(1));
        assertEquals(3 * 4 * 5, shape.size(2));
    }

    @Test
    void testStrideComposition() {
        // Stride.of(100, Stride.of(10, 1)) → (100, (10, 1))
        Stride stride = Stride.of(100, Stride.of(10, 1));

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());

        System.out.println("Stride nested: " + stride);
    }

    @Test
    void testStrideSingleElementNormalized() {
        // Stride.of(Stride.of(100), Stride.of(10, 1)) → (100, (10, 1))
        Stride stride = Stride.of(Stride.of(100), Stride.of(10, 1));

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testStrideUnwrapSingle() {
        // Stride.of(Stride.of(10, 1)) → (10, 1)
        Stride inner = Stride.of(10, 1);
        Stride stride = Stride.of(inner);

        assertSame(inner, stride);
        assertTrue(stride.isFlat());
    }

    @Test
    void testSingleNumber() {
        // Shape.of(5) → (5)
        Shape shape = Shape.of(5);

        assertEquals(1, shape.rank());
        assertEquals(1, shape.flatRank());
        assertArrayEquals(new long[] {5}, shape.toArray());
    }

    @Test
    void testNestedPreservesInternalStructure() {
        // Create a nested shape (2, (3, 4))
        Shape nested = Shape.of(2, Shape.of(3, 4));

        // Compose it: Shape.of(1, nested, 5) → (1, (2, (3, 4)), 5)
        Shape composed = Shape.of(1, nested, 5);

        assertEquals(3, composed.rank());
        assertEquals(5, composed.flatRank());
        assertArrayEquals(new long[] {1, 2, 3, 4, 5}, composed.toArray());

        System.out.println("Composed with nested: " + composed);

        // Mode 1 should preserve the nested structure (2, (3, 4))
        Shape mode1 = composed.modeAt(1);
        assertEquals(2, mode1.rank());
        assertEquals(3, mode1.flatRank());
    }

    @Test
    void testInvalidArguments() {
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Shape.of("invalid");
                });

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.of("invalid");
                });
    }
}
