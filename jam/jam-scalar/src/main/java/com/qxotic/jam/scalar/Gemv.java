package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.qxotic.jam.JAM.Parallel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * The decode kernel ({@code n == 1}): one dot product per weight row, rows claimed in small runs by
 * the workers. Each row is decoded a {@link #CHUNK}-element span at a time into an L1-resident
 * buffer and folded into a running vector of partial sums, {@code acc[i] = fma(tmp[i], x[i],
 * acc[i])} with the activation held as one {@code float[]} per chunk - every array indexed alike,
 * so both JITs vectorize the fold and the row costs what its decode costs. The partial sums are
 * summed once per row.
 */
final class Gemv {

    private Gemv() {}

    /** Elements decoded per step (a multiple of every block size). */
    static final int CHUNK = 256;

    /** Rows claimed per step of the dynamic split. */
    static final int ROWS = 16;

    static void run(
            Parallel parallel,
            Scratch scratch,
            Weight w,
            MemorySegment a,
            long aOff,
            MemorySegment r,
            long rOff,
            int m,
            int k) {
        int chunks = Tile.ceilDiv(k, CHUNK);
        float[][] x = new float[chunks][CHUNK];
        for (int c = 0; c < chunks; c++) {
            int len = Math.min(CHUNK, k - c * CHUNK);
            MemorySegment.copy(a, JAVA_FLOAT_UNALIGNED, aOff + c * CHUNK * 4L, x[c], 0, len);
        }
        parallel.forLoop(
                Tile.ceilDiv(m, ROWS),
                (strip, slot) -> {
                    Scratch.Slot s = scratch.slot(slot);
                    float[] tmp = s.row(CHUNK), acc = s.acc(CHUNK);
                    for (int i = strip * ROWS, to = Math.min(m, i + ROWS); i < to; i++)
                        r.set(
                                JAVA_FLOAT_UNALIGNED,
                                rOff + i * 4L,
                                dot(w, i, s.decode, x, k, tmp, acc));
                });
    }

    /**
     * {@code acc[i] += w[i] x[i]} over one chunk: every array indexed by the bare {@code i} and an
     * exact trip count, the shape both JITs vectorize at full width (see {@link Tile}).
     */
    private static void fold(float[] w, float[] x, float[] acc) {
        for (int i = 0; i < CHUNK; i++) acc[i] = Math.fma(w[i], x[i], acc[i]);
    }

    private static float dot(
            Weight w, int row, Decode decode, float[][] x, int k, float[] tmp, float[] acc) {
        Arrays.fill(acc, 0f);
        for (int c = 0, l0 = 0; l0 < k; c++, l0 += CHUNK) {
            int len = Math.min(CHUNK, k - l0);
            decode.row(w, row, l0, len, CHUNK, tmp, 0);
            fold(tmp, x[c], acc);
        }
        float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
        for (int i = 0; i < CHUNK; i += 4) {
            s0 += acc[i];
            s1 += acc[i + 1];
            s2 += acc[i + 2];
            s3 += acc[i + 3];
        }
        return (s0 + s1) + (s2 + s3);
    }
}
