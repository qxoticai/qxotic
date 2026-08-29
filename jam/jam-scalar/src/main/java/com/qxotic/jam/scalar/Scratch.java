package com.qxotic.jam.scalar;

import java.util.Arrays;

/**
 * The buffers of one {@link ScalarJAM}: owned by the provider instance and passed into every kernel
 * - not a {@code static}, not a {@code ThreadLocal} (the jam-vector {@code Scratch} idiom: buffers
 * rooted in JVM-lifetime worker threads or in a class outlive the context that used them, and a
 * per-instance lock cannot guard class-global state). Everything here is reachable only through the
 * provider, reused across calls with no steady-state allocation, and collected with it.
 *
 * <p>Two kinds: the panels a whole call shares, and one {@link Slot} per task of a region - a
 * kernel asks the host for at most {@code width} tasks and the task index is unique among those
 * live at once, so a slot is never used by two threads at once and needs no free list.
 */
final class Scratch {

    /** {@link Gemm}: the transposed activation blocks, one panel per k-block. */
    Rows[] panels = new Rows[0];

    /** {@link RowGemm}: the activation copy (rows of {@code k + KU}) and the split partials. */
    float[][] activation = new float[0][];

    float[] partials = new float[0];

    private final Slot[] slots;

    Scratch(int width) {
        slots = new Slot[width];
    }

    Slot slot(int i) {
        Slot s = slots[i];
        if (s == null) slots[i] = s = new Slot();
        return s;
    }

    /** {@link Gemm}'s panel for k-block {@code kb}, fitted to {@code rows x len}. */
    Rows panel(int kb, int rows, int len) {
        if (panels.length <= kb) panels = Arrays.copyOf(panels, kb + 1);
        if (panels[kb] == null) panels[kb] = new Rows();
        return panels[kb].fit(rows, len);
    }

    /** One worker's buffers; each kernel uses the members it needs, allocated on first use. */
    static final class Slot {
        final Decode decode = new Decode();

        /** The tile's three scalar rows at stride {@link Tile#XS}. */
        final float[] x = new float[Tile.TR * Tile.XS];

        /** Accumulator rows: {@link Gemm}'s group of rows, {@link RowGemm}'s tokens. */
        final Rows c = new Rows();

        /** {@link RowGemm}: the transposed weight band. */
        final Rows wT = new Rows();

        /**
         * A decoded row ({@link RowGemm}, {@link Gemv}) or packed activation row ({@link Gemm}).
         */
        float[] row = new float[0];

        /** {@link Gemv}: the running partial sums; {@link RowGemm}: the reduced result row. */
        float[] acc = new float[0];

        float[] row(int len) {
            if (row.length < len) row = new float[len];
            return row;
        }

        float[] acc(int len) {
            if (acc.length < len) acc = new float[len];
            return acc;
        }
    }
}
