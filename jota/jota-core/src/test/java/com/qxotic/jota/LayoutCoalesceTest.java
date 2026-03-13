package com.qxotic.jota;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;

class LayoutCoalesceTest {

    @Test
    void coalesce_full_rowMajorCollapsesToSingleMode() {
        Layout layout = Layout.rowMajor(2, 3, 4);

        Layout coalesced = layout.coalesce();

        assertEquals(1, coalesced.shape().rank());
        assertArrayEquals(new long[] {24}, coalesced.shape().toArray());
        assertArrayEquals(new long[] {1}, coalesced.stride().toArray());
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void coalesce_full_removesSingletonModes() {
        Layout layout = Layout.of(Shape.flat(2, 1, 3), Stride.flat(3, 99, 1));

        Layout coalesced = layout.coalesce();

        assertEquals(Layout.of(Shape.flat(6), Stride.flat(1)), coalesced);
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void coalesce_full_nonMergeableIsNoopInValue() {
        Layout layout = Layout.of(Shape.flat(2, 3, 4), Stride.flat(100, 10, 1));

        Layout coalesced = layout.coalesce();

        assertEquals(layout.flatten(), coalesced);
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void coalesce_full_negativeStrideCanMerge() {
        Layout layout = Layout.of(Shape.flat(2, 3), Stride.flat(-3, -1));

        Layout coalesced = layout.coalesce();

        assertEquals(Layout.of(Shape.flat(6), Stride.flat(-1)), coalesced);
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void coalesce_full_scalarReturnsSelf() {
        Layout scalar = Layout.scalar();
        assertSame(scalar, scalar.coalesce());
    }

    @Test
    void coalesce_full_isIdempotent() {
        Layout layout = Layout.of(Shape.flat(2, 3, 4), Stride.flat(12, 4, 1));

        Layout once = layout.coalesce();
        Layout twice = once.coalesce();

        assertEquals(once, twice);
    }

    @Test
    void coalesce_mode_mergesSelectedPair() {
        Layout layout = Layout.rowMajor(2, 3, 4);

        Layout coalesced = layout.coalesce(1);

        assertEquals(Layout.of(Shape.flat(2, 12), Stride.flat(12, 1)), coalesced);
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void coalesce_mode_nonMergeableReturnsSameInstance() {
        Layout layout = Layout.of(Shape.flat(2, 3, 4), Stride.flat(100, 10, 1));

        Layout coalesced = layout.coalesce(0);

        assertSame(layout, coalesced);
    }

    @Test
    void coalesce_mode_lastModeReturnsSameInstance() {
        Layout layout = Layout.rowMajor(2, 3, 4);

        assertSame(layout, layout.coalesce(-1));
        assertSame(layout, layout.coalesce(2));
    }

    @Test
    void coalesce_mode_rankOneReturnsSameInstance() {
        Layout layout = Layout.of(Shape.flat(8), Stride.flat(2));
        assertSame(layout, layout.coalesce(0));
    }

    @Test
    void coalesce_mode_nestedSelectedPairCanMerge() {
        Shape shape = Shape.of(2, Shape.of(3L, 4L), 5L);
        Layout layout = Layout.rowMajor(shape);

        Layout coalesced = layout.coalesce(0);

        assertEquals(2, coalesced.shape().rank());
        assertArrayEquals(new long[] {24, 5}, coalesced.shape().toArray());
        assertArrayEquals(new long[] {5, 1}, coalesced.stride().toArray());
        assertOneDimensionalMappingPreserved(layout, coalesced);
    }

    @Test
    void canCoalesce_flatLayoutFastPath() {
        Layout rowMajor = Layout.rowMajor(2, 3, 4);
        assertTrue(rowMajor.canCoalesce(0));
        assertTrue(rowMajor.canCoalesce(1));
        assertFalse(rowMajor.canCoalesce(-1));

        Layout nonMergeable = Layout.of(Shape.flat(2, 3, 4), Stride.flat(100, 10, 1));
        assertFalse(nonMergeable.canCoalesce(0));
        assertFalse(nonMergeable.canCoalesce(1));
    }

    private static void assertOneDimensionalMappingPreserved(Layout original, Layout transformed) {
        assertEquals(original.shape().size(), transformed.shape().size());
        long total = original.shape().size();
        for (long i = 0; i < total; i++) {
            assertEquals(
                    evaluateAtNaturalIndex(original, i), evaluateAtNaturalIndex(transformed, i));
        }
    }

    private static long evaluateAtNaturalIndex(Layout layout, long linear) {
        long[] coord = Indexing.linearToCoord(layout.shape(), linear);
        return Indexing.coordToOffset(layout.stride(), coord);
    }
}
