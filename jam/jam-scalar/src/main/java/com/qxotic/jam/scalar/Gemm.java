package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.qxotic.jam.JAM.Parallel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicIntegerArray;

/**
 * The prefill kernel ({@code n >= }{@link #MIN_N}): {@code C[t][i] = sum_l W[i][l] A[t][l]} with
 * the token axis as the SIMD axis of the {@link Tile}. A token block of the activation is
 * transposed once into panel rows {@code aT[l][t]} (the k-block sized for L2), each weight row is
 * decoded once per k-block into a {@code float[]} of broadcast scalars, and a work item is a group
 * of rows over all of k: its accumulator rows live in per-worker scratch across the k-blocks and
 * are transposed into the token-major result at the end. No dtype needs more than a row decoder.
 *
 * <p>One region per token block: the k-block packs are the first items and the row groups the rest,
 * pulled from a counter so the tail balances; a group spins on a block's flag before its first use
 * of it, and since every pack is claimed before any group starts, that wait is at most one pack in
 * flight.
 */
final class Gemm {

    private Gemm() {}

    /** Smallest batch this kernel serves; below it the token loop is too short for the JITs. */
    static final int MIN_N = 256;

    /** Tokens per block: the tile's vector length. */
    static final int NB = 512;

    /** Target footprint of one transposed activation block ({@code kc x nb} floats). */
    static final int PANEL_BYTES = 512 << 10;

    /** Largest k-block (at the smallest batch). */
    static final int KC_MAX = PANEL_BYTES / (4 * MIN_N);

    /** Groups per worker: enough items for the dynamic tail to balance. */
    static final int GROUPS_PER_WORKER = 4;

    /** Most rows in one work item (a multiple of {@link Tile#TR}). */
    static final int G_MAX = 8 * Tile.TR;

    static void run(
            Parallel parallel,
            Scratch scratch,
            Weight w,
            MemorySegment a,
            long aOff,
            int lda,
            MemorySegment r,
            long rOff,
            int ldr,
            int m,
            int n,
            int k) {
        int epb = w.blockElems();
        int nb = Tile.ceilDiv(n, Tile.ceilDiv(n, NB)); // balanced blocks: none under NB / 2
        int kc = Math.min(Math.min(k, KC_MAX), Math.max(epb, PANEL_BYTES / (4 * nb) / epb * epb));
        int kcPad = Tile.padK(kc);
        int kBlocks = Tile.ceilDiv(k, kc);
        int workers = parallel.width();
        int g =
                Tile.TR
                        * Math.clamp(
                                Tile.ceilDiv(m, Tile.TR * workers * GROUPS_PER_WORKER),
                                1,
                                G_MAX / Tile.TR);
        int groups = Tile.ceilDiv(m, g);

        for (int t0 = 0; t0 < n; t0 += nb) {
            int tokens = Math.min(nb, n - t0), lanes = Tile.lanes(tokens), base = t0;
            Rows[] aT = new Rows[kBlocks];
            for (int kb = 0; kb < kBlocks; kb++) aT[kb] = scratch.panel(kb, kcPad, lanes);
            AtomicIntegerArray packed = new AtomicIntegerArray(kBlocks);
            AtomicInteger next = new AtomicInteger();
            AtomicBoolean failed = new AtomicBoolean();
            parallel.forLoop(
                    Math.min(kBlocks + groups, workers),
                    worker -> {
                        Scratch.Slot s = scratch.slot(worker);
                        for (int item; (item = next.getAndIncrement()) < kBlocks + groups; ) {
                            if (item < kBlocks) {
                                int l0 = item * kc, len = Math.min(kc, k - l0);
                                long src = aOff + ((long) base * lda + l0) * 4;
                                try {
                                    pack(a, src, lda, tokens, lanes, len, kcPad, aT[item], s);
                                } catch (Throwable t) {
                                    failed.set(true); // groups waiting on this block give up
                                    throw t;
                                }
                                packed.set(item, 1);
                                continue;
                            }
                            int row0 = (item - kBlocks) * g, rows = Math.min(g, m - row0);
                            Rows c = group(w, row0, rows, k, kc, aT, packed, failed, lanes, s);
                            long dst = rOff + ((long) base * ldr + row0) * 4;
                            for (int i = 0; i < rows; i++)
                                store(c.row(i), tokens, r, dst + i * 4L, ldr);
                        }
                    });
        }
    }

    /**
     * Transposes a token block of the activation: {@code aT[l][t] = A[t][l0 + l]}, zero past the
     * block's k and past its tokens up to the lane width.
     */
    private static void pack(
            MemorySegment a,
            long src,
            int lda,
            int tokens,
            int lanes,
            int len,
            int kcPad,
            Rows aT,
            Scratch.Slot s) {
        float[] row = s.row(len);
        for (int t = 0; t < tokens; t++, src += lda * 4L) {
            MemorySegment.copy(a, JAVA_FLOAT_UNALIGNED, src, row, 0, len);
            for (int l = 0; l < len; l++) aT.row(l)[t] = row[l];
            for (int l = len; l < kcPad; l++) aT.row(l)[t] = 0f;
        }
        for (int l = 0; l < kcPad; l++) Arrays.fill(aT.row(l), tokens, lanes, 0f);
    }

    /** One work item: rows {@code row0 .. row0+rows} over all of k, into the slot's c rows. */
    private static Rows group(
            Weight w,
            int row0,
            int rows,
            int k,
            int kc,
            Rows[] aT,
            AtomicIntegerArray packed,
            AtomicBoolean failed,
            int lanes,
            Scratch.Slot s) {
        Rows c = s.c.fit(G_MAX, NB);
        c.zero(0, rows, lanes);
        for (int kb = 0, l0 = 0; l0 < k; kb++, l0 += kc) {
            int len = Math.min(kc, k - l0), lenPad = Tile.padK(len);
            while (packed.get(kb) == 0) {
                if (failed.get()) return c;
                Thread.onSpinWait();
            }
            float[][] q = aT[kb].rows();
            for (int i = 0; i < rows; i += Tile.TR) {
                for (int j = 0; j < Tile.TR; j++) {
                    if (i + j < rows)
                        s.decode.row(w, row0 + i + j, l0, len, lenPad, s.x, j * Tile.XS);
                    else Arrays.fill(s.x, j * Tile.XS, j * Tile.XS + lenPad, 0f);
                }
                Tile.sweep(q, lenPad, s.x, c.row(i), c.row(i + 1), c.row(i + 2), lanes);
            }
        }
        return c;
    }

    /** Writes one accumulator row into the token-major result: {@code R[t][i]}. */
    private static void store(float[] c, int tokens, MemorySegment r, long dst, int ldr) {
        for (int t = 0; t < tokens; t++)
            r.set(JAVA_FLOAT_UNALIGNED, dst + (long) t * ldr * 4, c[t]);
    }
}
