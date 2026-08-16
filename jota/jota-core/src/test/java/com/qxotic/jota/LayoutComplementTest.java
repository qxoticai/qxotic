package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class LayoutComplementTest {

    @Test
    void complement_basicOneDimensionalFixture() {
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(1));

        Layout complement = layout.complement(24);

        assertEquals(Layout.of(Shape.flat(6), Stride.flat(4)), complement);
    }

    @Test
    void complement_inverseFixture() {
        Layout layout = Layout.of(Shape.flat(6), Stride.flat(4));

        Layout complement = layout.complement(24);

        assertEquals(Layout.of(Shape.flat(4), Stride.flat(1)), complement);
    }

    @Test
    void complement_strideGapFixture() {
        Layout layout = Layout.of(Shape.flat(2, 4), Stride.flat(1, 6));

        Layout complement = layout.complement(24);

        assertEquals(Layout.of(Shape.flat(3), Stride.flat(2)), complement);
    }

    @Test
    void complement_scalarOrDegenerateGivesRowMajorOfCotarget() {
        assertEquals(Layout.rowMajor(7), Layout.scalar().complement(7));

        Layout singleton = Layout.of(Shape.flat(1, 1), Stride.flat(10, 11));
        assertEquals(Layout.rowMajor(5), singleton.complement(5));
    }

    @Test
    void complement_zeroSizeLayoutGivesRowMajorOfCotarget() {
        Layout zero = Layout.of(Shape.flat(0, 3), Stride.flat(5, 1));
        assertEquals(Layout.rowMajor(9), zero.complement(9));
    }

    @Test
    void complement_isDeterministicAndCanonical() {
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));

        Layout c1 = layout.complement(24);
        Layout c2 = layout.complement(24);

        assertEquals(c1, c2);
        assertEquals(c1, c1.coalesce());
    }

    @Test
    void complement_coverageAtLeastCotarget() {
        Layout layout = Layout.of(Shape.flat(4), Stride.flat(2));

        Layout complement = layout.complement(24);
        Layout paired = makeLayout(layout, complement);

        assertTrue(paired.cosize() >= 24);
    }

    @Test
    void complement_rejectsNonInjectiveLayout() {
        Layout broadcast = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> broadcast.complement(8));
        assertTrue(ex.getMessage().contains("injective"));
    }

    @Test
    void complement_rejectsNegativeStrideLayout() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(-3, -1));
        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> layout.complement(8));
        assertTrue(ex.getMessage().contains("non-negative"));
    }

    @Test
    void complement_rejectsNonDivisibleStrideChain() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(2, 3));
        assertThrows(IllegalArgumentException.class, () -> layout.complement(16));
    }

    @Test
    void complement_rejectsInvalidCotarget() {
        Layout layout = Layout.rowMajor(4);
        assertThrows(IllegalArgumentException.class, () -> layout.complement(0));
        assertThrows(IllegalArgumentException.class, () -> layout.complement(-1));
    }

    @Test
    void complement_overflowPropagatesArithmeticException() {
        Layout hugeStride = Layout.of(Shape.flat(2), Stride.flat(Long.MAX_VALUE));
        assertThrows(ArithmeticException.class, () -> hugeStride.complement(Integer.MAX_VALUE));
    }

    private static Layout makeLayout(Layout left, Layout right) {
        return Layout.of(
                Shape.of(left.shape(), right.shape()), Stride.of(left.stride(), right.stride()));
    }
}
