package ai.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for Stride methods including rank, modeAt, flatten, rowMajor, columnMajor,
 * etc.
 */
class StrideMethodsTest {

    @Test
    void testFlatStrideBasics() {
        Stride stride = Stride.flat(6, 3, 1);

        assertEquals(3, stride.rank());
        assertEquals(3, stride.flatRank());
        assertTrue(stride.isFlat());
        assertFalse(stride.isScalar());
        assertArrayEquals(new long[] {6, 3, 1}, stride.toArray());
    }

    @Test
    void testNestedStrideBasics() {
        // (12, (4, 1))
        Stride stride = Stride.of(12, Stride.of(4L, 1L));

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertFalse(stride.isFlat());
        assertFalse(stride.isScalar());
        assertArrayEquals(new long[] {12, 4, 1}, stride.toArray());
    }

    @Test
    void testScalarStride() {
        Stride scalar = Stride.scalar();

        assertEquals(0, scalar.rank());
        assertEquals(0, scalar.flatRank());
        assertTrue(scalar.isFlat());
        assertTrue(scalar.isScalar());
        assertArrayEquals(new long[] {}, scalar.toArray());
    }

    @Test
    void testFlatAtPositiveIndices() {
        Stride stride = Stride.flat(12, 4, 1);

        assertEquals(12, stride.flatAt(0));
        assertEquals(4, stride.flatAt(1));
        assertEquals(1, stride.flatAt(2));
    }

    @Test
    void testFlatAtNegativeIndices() {
        Stride stride = Stride.flat(12, 4, 1);

        assertEquals(1, stride.flatAt(-1)); // Last
        assertEquals(4, stride.flatAt(-2)); // Second to last
        assertEquals(12, stride.flatAt(-3)); // Third to last
    }

    @Test
    void testModeAtFlatStride() {
        Stride stride = Stride.flat(12, 4, 1);

        Stride mode0 = stride.modeAt(0);
        assertEquals(1, mode0.rank());
        assertEquals(12, mode0.flatAt(0));

        Stride mode1 = stride.modeAt(1);
        assertEquals(1, mode1.rank());
        assertEquals(4, mode1.flatAt(0));

        Stride mode2 = stride.modeAt(2);
        assertEquals(1, mode2.rank());
        assertEquals(1, mode2.flatAt(0));
    }

    @Test
    void testModeAtNestedStride() {
        // (12, (4, 1))
        Stride stride = Stride.of(12, Stride.of(4L, 1L));

        Stride mode0 = stride.modeAt(0);
        assertEquals(1, mode0.rank());
        assertTrue(mode0.isFlat());
        assertEquals(12, mode0.flatAt(0));

        Stride mode1 = stride.modeAt(1);
        assertEquals(2, mode1.rank());
        assertTrue(mode1.isFlat());
        assertArrayEquals(new long[] {4, 1}, mode1.toArray());
    }

    @Test
    void testModeAtNegativeIndex() {
        Stride stride = Stride.flat(12, 4, 1);

        Stride lastMode = stride.modeAt(-1);
        assertEquals(1, lastMode.rank());
        assertEquals(1, lastMode.flatAt(0));

        Stride secondLastMode = stride.modeAt(-2);
        assertEquals(1, secondLastMode.rank());
        assertEquals(4, secondLastMode.flatAt(0));
    }

    @Test
    void testFlattenFlatStride() {
        Stride stride = Stride.flat(12, 4, 1);
        Stride flattened = stride.flatten();

        assertSame(stride, flattened); // Should return same instance
    }

    @Test
    void testFlattenNestedStride() {
        // (12, (4, 1))
        Stride stride = Stride.of(12, Stride.of(4L, 1L));
        Stride flattened = stride.flatten();

        assertTrue(flattened.isFlat());
        assertEquals(3, flattened.rank());
        assertEquals(3, flattened.flatRank());
        assertArrayEquals(new long[] {12, 4, 1}, flattened.toArray());
    }

    @Test
    void testRowMajorFlat() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.rowMajor(shape);

        assertTrue(stride.isFlat());
        assertEquals(3, stride.rank());
        assertArrayEquals(new long[] {12, 4, 1}, stride.toArray());
    }

    @Test
    void testColumnMajorFlat() {
        Shape shape = Shape.flat(2, 3, 4);
        Stride stride = Stride.columnMajor(shape);

        assertTrue(stride.isFlat());
        assertEquals(3, stride.rank());
        assertArrayEquals(new long[] {1, 2, 6}, stride.toArray());
    }

    @Test
    void testRowMajorNested() {
        // Shape: (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.rowMajor(shape);

        assertFalse(stride.isFlat());
        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {12, 4, 1}, stride.toArray());
    }

    @Test
    void testColumnMajorNested() {
        // Shape: (2, (3, 4))
        Shape shape = Shape.of(2, Shape.of(3L, 4L));
        Stride stride = Stride.columnMajor(shape);

        assertFalse(stride.isFlat());
        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {1, 2, 6}, stride.toArray());
    }

    @Test
    void testRowMajorScalar() {
        Shape scalar = Shape.scalar();
        Stride stride = Stride.rowMajor(scalar);

        assertTrue(stride.isScalar());
        assertEquals(0, stride.rank());
        assertArrayEquals(new long[] {}, stride.toArray());
    }

    @Test
    void testColumnMajorScalar() {
        Shape scalar = Shape.scalar();
        Stride stride = Stride.columnMajor(scalar);

        assertTrue(stride.isScalar());
        assertEquals(0, stride.rank());
        assertArrayEquals(new long[] {}, stride.toArray());
    }

    @Test
    void testTemplateBasedStride() {
        Shape template = Shape.pattern("(_,(_,_))", 2, 3, 4);
        Stride stride = Stride.template(template, 100, 10, 1);

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testDeeplyNestedStride() {
        // (100, (10, (2, 1)))
        Stride stride = Stride.of(100, Stride.of(10, Stride.of(2L, 1L)));

        assertEquals(2, stride.rank());
        assertEquals(4, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 2, 1}, stride.toArray());

        // Test mode extraction
        Stride mode1 = stride.modeAt(1);
        assertEquals(2, mode1.rank());
        assertEquals(3, mode1.flatRank());
        assertArrayEquals(new long[] {10, 2, 1}, mode1.toArray());
    }

    @Test
    void testRowMajorPreservesNesting() {
        Shape shape = Shape.pattern("((N, M), K)", 2, 3, 4);
        Stride stride = Stride.rowMajor(shape);

        assertEquals(shape.rank(), stride.rank());
        assertEquals(shape.flatRank(), stride.flatRank());
        assertArrayEquals(new long[] {12, 4, 1}, stride.toArray());
    }

    @Test
    void testColumnMajorPreservesNesting() {
        Shape shape = Shape.pattern("((N, M), K)", 2, 3, 4);
        Stride stride = Stride.columnMajor(shape);

        assertEquals(shape.rank(), stride.rank());
        assertEquals(shape.flatRank(), stride.flatRank());
        assertArrayEquals(new long[] {1, 2, 6}, stride.toArray());
    }

    @Test
    void testToString() {
        Stride flat = Stride.flat(12, 4, 1);
        assertEquals("(12, 4, 1)", flat.toString());

        Stride nested = Stride.of(12, Stride.of(4L, 1L));
        assertEquals("(12, (4, 1))", nested.toString());
    }

    @Test
    void testComposition() {
        // Stride.of(100, Stride.of(10, 1)) → (100, (10, 1))
        Stride stride = Stride.of(100, Stride.of(10L, 1L));

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testSingleElementNormalization() {
        // Single-element strides get unwrapped: nested(100, Stride.of(10), Stride.of(1)) → (100,
        // 10, 1)
        Stride stride = Stride.of(100, Stride.of(10L), Stride.of(1L));

        assertTrue(stride.isFlat());
        assertEquals(3, stride.rank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testUnwrapSingle() {
        // nested(Stride.of(12, 4)) → (12, 4)
        Stride inner = Stride.of(12L, 4L);
        Stride stride = Stride.of(inner);

        assertSame(inner, stride);
    }

    @Test
    void testStridePattern() {
        // Test flat pattern (a, b, c)
        Stride stride = Stride.pattern("(a, b, c)", 12, 4, 1);

        assertEquals(3, stride.rank());
        assertEquals(3, stride.flatRank());
        assertTrue(stride.isFlat());
        assertArrayEquals(new long[] {12, 4, 1}, stride.toArray());
    }

    @Test
    void testStrideNestedPattern() {
        // Test nested pattern (s, (s1, s2))
        Stride stride = Stride.pattern("(s, (s1, s2))", 100, 10, 1);

        assertEquals(2, stride.rank());
        assertEquals(3, stride.flatRank());
        assertArrayEquals(new long[] {100, 10, 1}, stride.toArray());
    }

    @Test
    void testStrideScalarPattern() {
        // "()" means scalar/empty nesting (no strides)
        Stride stride = Stride.pattern("()");

        assertEquals(0, stride.rank());
        assertEquals(0, stride.flatRank());
        assertArrayEquals(new long[] {}, stride.toArray());
    }

    @Test
    void testStrideSingletonPattern() {
        // "(_)" or "(stride)" means a singleton with one stride value
        Stride stride1 = Stride.pattern("(_)", 12);
        Stride stride2 = Stride.pattern("(stride)", 12);

        assertEquals(1, stride1.rank());
        assertEquals(1, stride1.flatRank());
        assertArrayEquals(new long[] {12}, stride1.toArray());

        assertEquals(1, stride2.rank());
        assertEquals(1, stride2.flatRank());
        assertArrayEquals(new long[] {12}, stride2.toArray());
    }

    @Test
    void testStrideEmptyIdentifiersNotAllowed() {
        // Empty identifiers should throw an exception
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(,)", 12, 4);
                },
                "Empty identifiers should not be allowed");

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(, (, ))", 100, 10, 1);
                },
                "Empty identifiers should not be allowed");

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(, (, ))", 100, 10, 1);
                },
                "Empty identifiers should not be allowed");
    }

    @Test
    void testStrideMalformedPatterns() {
        // Empty nested brackets (()) are not allowed
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(())");
                },
                "Empty nested brackets should not be allowed");

        // Empty nested brackets as part of a sequence
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("((), _)", 12);
                },
                "Empty nested brackets in sequence should not be allowed");

        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(_, ())", 12);
                },
                "Empty nested brackets in sequence should not be allowed");
    }

    @Test
    void testStrideNonNormalizedPatterns() {
        // Single-element nested brackets (( _ )) are not normalized
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("((_))", 12);
                },
                "Single-element nested brackets should be rejected as non-normalized");

        // ((s)) should also be rejected
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("((stride))", 12);
                },
                "Single-element nested brackets should be rejected as non-normalized");

        // More deeply nested single elements
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(((_)))", 12);
                },
                "Deeply nested single elements should be rejected");

        // Single element nested within a valid structure
        assertThrows(
                IllegalArgumentException.class,
                () -> {
                    Stride.pattern("(s1, ((s2)))", 12, 4);
                },
                "Single-element nested brackets in sequence should be rejected");
    }
}
