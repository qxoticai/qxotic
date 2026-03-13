package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Test;

class LayoutAlgebraTest {

    @Test
    void isInjectiveAndBijective_rowMajor_true() {
        Layout layout = Layout.rowMajor(2, 3, 4);
        assertEquals(isInjectiveOracle(layout), layout.isInjective());
        assertEquals(isBijectiveOracle(layout), layout.isBijective());
        assertTrue(layout.isInjective());
        assertTrue(layout.isBijective());
    }

    @Test
    void isInjectiveAndBijective_spanContiguousNonRowMajor_true() {
        Layout layout = Layout.of(Shape.flat(2, 2, 2), Stride.flat(4, 1, 2));
        assertEquals(isInjectiveOracle(layout), layout.isInjective());
        assertEquals(isBijectiveOracle(layout), layout.isBijective());
        assertTrue(layout.isInjective());
        assertTrue(layout.isBijective());
    }

    @Test
    void isInjectiveAndBijective_broadcast_false() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));
        assertEquals(isInjectiveOracle(layout), layout.isInjective());
        assertEquals(isBijectiveOracle(layout), layout.isBijective());
        assertFalse(layout.isInjective());
        assertFalse(layout.isBijective());
    }

    @Test
    void isInjectiveAndBijective_holey_falseForBijective() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        assertEquals(isInjectiveOracle(layout), layout.isInjective());
        assertEquals(isBijectiveOracle(layout), layout.isBijective());
        assertTrue(layout.isInjective());
        assertFalse(layout.isBijective());
    }

    @Test
    void isInjectiveAndBijective_zeroSize_trueByConvention() {
        Layout layout = Layout.of(Shape.flat(0, 3), Stride.flat(1, 1));
        assertEquals(isInjectiveOracle(layout), layout.isInjective());
        assertEquals(isBijectiveOracle(layout), layout.isBijective());
        assertTrue(layout.isInjective());
        assertTrue(layout.isBijective());
    }

    @Test
    void inverse_rowMajor_roundTrip() {
        Layout layout = Layout.rowMajor(2, 3, 4);
        Layout inverse = layout.inverse();
        Layout identity = Layout.rowMajor(layout.shape());

        assertEquals(identity, inverse.compose(layout));
        assertEquals(identity, layout.compose(inverse));
    }

    @Test
    void inverse_nonInjective_throws() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(0, 1));
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, layout::inverse);
        assertTrue(ex.getMessage().contains("bijective"));
    }

    @Test
    void inverse_injectiveNotBijective_throws() {
        Layout layout = Layout.of(Shape.flat(2, 2), Stride.flat(3, 1));
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, layout::inverse);
        assertTrue(ex.getMessage().contains("bijective"));
    }

    @Test
    void inverse_bijectiveButNonRepresentable_throws() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(1, 2));
        IllegalArgumentException ex = assertThrows(IllegalArgumentException.class, layout::inverse);
        assertTrue(ex.getMessage().contains("non-representable"));
    }

    @Test
    void compose_identity_leftAndRight() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(1, 2));
        Layout leftIdentity = Layout.rowMajor(layout.shape());

        assertEquals(layout, leftIdentity.compose(layout));
        assertEquals(layout, layout.compose(leftIdentity));
    }

    @Test
    void compose_associative_whenRepresentable() {
        Layout a = Layout.of(Shape.flat(2, 3), Stride.flat(12, 4));
        Layout b = Layout.of(Shape.flat(2, 2), Stride.flat(2, 1));
        Layout c = Layout.of(Shape.flat(2, 2), Stride.flat(1, 1));

        Layout left = a.compose(b).compose(c);
        Layout right = a.compose(b.compose(c));
        assertEquals(left, right);
    }

    @Test
    void compose_domainMismatch_throws() {
        Layout outer = Layout.of(Shape.flat(2, 3), Stride.flat(1, 2));
        Layout inner = Layout.rowMajor(7);

        assertThrows(IllegalArgumentException.class, () -> outer.compose(inner));
    }

    @Test
    void compose_nonRepresentable_throws() {
        Layout outer = Layout.of(Shape.flat(2, 3), Stride.flat(1, 2));
        Layout inner = Layout.rowMajor(6);

        IllegalArgumentException ex =
                assertThrows(IllegalArgumentException.class, () -> outer.compose(inner));
        assertTrue(ex.getMessage().contains("non-representable"));
    }

    @Test
    void compose_matchesSemanticOracle_onRepresentableCase() {
        Layout outer = Layout.of(Shape.flat(2, 3), Stride.flat(12, 4));
        Layout inner = Layout.of(Shape.flat(2, 2), Stride.flat(2, 1));
        Layout composed = outer.compose(inner);

        long[] dims = inner.shape().toArray();
        long[] coord = new long[dims.length];
        while (true) {
            long expected = composeSemanticValue(outer, inner, coord);
            long actual = Indexing.coordToOffset(composed.stride(), coord);
            assertEquals(expected, actual);
            if (!increment(coord, dims)) {
                break;
            }
        }
    }

    private static boolean isInjectiveOracle(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return true;
        }
        Set<Long> seen = new HashSet<>();
        long[] dims = layout.shape().toArray();
        long[] coord = new long[dims.length];
        while (true) {
            long offset = Indexing.coordToOffset(layout.stride(), coord);
            if (!seen.add(offset)) {
                return false;
            }
            if (!increment(coord, dims)) {
                break;
            }
        }
        return true;
    }

    private static boolean isBijectiveOracle(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return true;
        }
        Set<Long> seen = new HashSet<>();
        long[] dims = layout.shape().toArray();
        long[] coord = new long[dims.length];
        long min = Long.MAX_VALUE;
        long max = Long.MIN_VALUE;
        while (true) {
            long offset = Indexing.coordToOffset(layout.stride(), coord);
            if (!seen.add(offset)) {
                return false;
            }
            min = Math.min(min, offset);
            max = Math.max(max, offset);
            if (!increment(coord, dims)) {
                break;
            }
        }
        long span = max - min + 1;
        return span == layout.shape().size();
    }

    private static long composeSemanticValue(Layout outer, Layout inner, long[] innerCoord) {
        long innerLinear = Indexing.coordToOffset(inner.stride(), innerCoord);
        long[] outerCoord = Indexing.linearToCoord(outer.shape(), innerLinear);
        return Indexing.coordToOffset(outer.stride(), outerCoord);
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
