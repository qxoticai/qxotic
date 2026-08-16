package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class LayoutCosizeTest {

    @Test
    void cosize_rowMajorEqualsSize() {
        Layout layout = Layout.rowMajor(2, 3, 4);

        assertEquals(24, layout.cosize());
        assertEquals(layout.shape().size(), layout.cosize());
    }

    @Test
    void cosize_scalarIsOne() {
        assertEquals(1, Layout.scalar().cosize());
    }

    @Test
    void cosize_zeroSizeIsZero() {
        Layout layout = Layout.of(Shape.flat(0, 3), Stride.flat(5, 1));
        assertEquals(0, layout.cosize());
    }

    @Test
    void cosize_holeyLayoutGreaterThanSize() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));

        assertEquals(5, layout.cosize());
        assertTrue(layout.cosize() > layout.shape().size());
    }

    @Test
    void cosize_broadcastCanBeSmallerThanSize() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));

        assertEquals(3, layout.cosize());
        assertTrue(layout.cosize() < layout.shape().size());
    }

    @Test
    void cosize_negativeStride() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(-3, -1));
        assertEquals(6, layout.cosize());
    }

    @Test
    void cosize_singletonAxesDoNotAffectSpan() {
        Layout withSingleton = Layout.of(Shape.flat(2, 1, 3), Stride.flat(3, 99, 1));
        Layout withoutSingleton = Layout.of(Shape.flat(2, 3), Stride.flat(3, 1));

        assertEquals(withoutSingleton.cosize(), withSingleton.cosize());
    }

    @Test
    void cosize_matchesEnumeratedMinMax() {
        Layout layout = Layout.of(Shape.flat(2, 3, 2), Stride.flat(7, -2, 5));
        assertEquals(cosizeOracle(layout), layout.cosize());
    }

    @Test
    void cosize_overflowThrowsArithmeticException() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(Long.MAX_VALUE, Long.MAX_VALUE));
        assertThrows(ArithmeticException.class, layout::cosize);
    }

    @Test
    void cosize_bijectiveImpliesEqualsSize_forKnownCases() {
        Layout rowMajor = Layout.rowMajor(2, 3, 4);
        Layout permutedButBijective = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));

        assertTrue(rowMajor.isBijective());
        assertTrue(permutedButBijective.isBijective());
        assertEquals(rowMajor.shape().size(), rowMajor.cosize());
        assertEquals(permutedButBijective.shape().size(), permutedButBijective.cosize());
    }

    private static long cosizeOracle(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return 0;
        }
        long[] dims = layout.shape().toArray();
        long[] coord = new long[dims.length];
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        while (true) {
            long offset = Indexing.coordToOffset(layout.stride(), coord);
            min = Math.min(min, offset);
            max = Math.max(max, offset);
            if (!increment(coord, dims)) {
                break;
            }
        }
        return (max - min) + 1;
    }

    private static boolean increment(long[] coord, long[] dims) {
        for (int i = dims.length - 1; i >= 0; i--) {
            coord[i]++;
            if (coord[i] < dims[i]) {
                return true;
            }
            coord[i] = 0;
        }
        return false;
    }
}
