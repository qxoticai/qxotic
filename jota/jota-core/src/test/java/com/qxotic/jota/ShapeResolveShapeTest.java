package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import org.junit.jupiter.api.Test;

class ShapeResolveShapeTest {

    @Test
    void resolveShapeWithoutInferenceRequiresExactSize() {
        Shape shape = Shape.resolveShape(12, 3, 4);
        assertEquals(2, shape.rank());
        assertArrayEquals(new long[] {3, 4}, shape.toArray());
    }

    @Test
    void resolveShapeInfersSingleMinusOne() {
        Shape shape = Shape.resolveShape(12, -1, 4);
        assertArrayEquals(new long[] {3, 4}, shape.toArray());

        Shape shapeMiddle = Shape.resolveShape(24, 2, -1, 3);
        assertArrayEquals(new long[] {2, 4, 3}, shapeMiddle.toArray());

        Shape shapeLast = Shape.resolveShape(24, 2, 3, -1);
        assertArrayEquals(new long[] {2, 3, 4}, shapeLast.toArray());
    }

    @Test
    void resolveShapeSingleDimensionExactMatch() {
        Shape shape = Shape.resolveShape(7, 7);
        assertArrayEquals(new long[] {7}, shape.toArray());
    }

    @Test
    void resolveShapeRejectsEmptyDims() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(1));
    }

    @Test
    void resolveShapeRejectsZeroDimensions() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(12, 0, 12));
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(12, 12, 0));
    }

    @Test
    void resolveShapeRejectsMultipleMinusOne() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(12, -1, -1));
    }

    @Test
    void resolveShapeRejectsLessThanMinusOne() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(12, -2, 3));
    }

    @Test
    void resolveShapeRejectsMismatchedKnownProduct() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(12, 2, 5));
    }

    @Test
    void resolveShapeRejectsNonDivisibleInference() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(10, 3, -1));
    }

    @Test
    void resolveShapeRejectsOverflowingKnownProduct() {
        assertThrows(ArithmeticException.class, () -> Shape.resolveShape(10, Long.MAX_VALUE, 2));
    }

    @Test
    void resolveShapeRejectsNullDims() {
        assertThrows(NullPointerException.class, () -> Shape.resolveShape(10, (long[]) null));
    }

    @Test
    void resolveShapeRejectsZeroTotalSizeInference() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(0, -1, 2));
    }

    @Test
    void resolveShapeRejectsNegativeTotalSize() {
        assertThrows(IllegalArgumentException.class, () -> Shape.resolveShape(-1, 1));
    }
}
