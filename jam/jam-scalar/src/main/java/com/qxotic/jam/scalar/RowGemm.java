package com.qxotic.jam.scalar;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.qxotic.jam.JAM.Parallel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;

/**
 * The small-batch prefill kernel ({@code 2 <= n < }{@link Gemm#MIN_N}): the {@link Tile} with the
 * axes swapped, the weight rows as the SIMD axis - a token loop shorter than a few hundred runs at
 * a quarter width or worse on C2, a band of {@link #MR} weight rows is always long enough.
 *
 * <p>A work item is a band of rows over one k-split: it decodes and transposes the band into panel
 * rows {@code wT[l][i]} (sized for L2), takes the activation scalars straight from a copy of the
 * token rows, sweeps three tokens per pass and stores its partial sums. Items are {@code bands x
 * k-splits} so there are enough of them for the pool; one reduce item per token, at the end of the
 * same region, sums the splits into the contiguous result row with one bulk copy.
 */
final class RowGemm {

    private RowGemm() {}

    /** Rows per band: the tile's vector length. */
    static final int MR = 512;

    /** k-steps per k-block (a multiple of every block size): the transposed band's L2 footprint. */
    static final int KC = 256;

    /** Items per worker: enough for the dynamic tail to balance. */
    static final int ITEMS_PER_WORKER = 4;

    /**
     * Largest split-partials buffer this kernel allocates; above it the caller uses {@link Gemm}.
     */
    static final long MAX_PARTIALS = 1L << 28;

    /** Whether the split partials of a {@code [m x n]} result over {@code k} fit the bound. */
    static boolean fits(int m, int n, int k, int width) {
        return (long) splits(m, k, width) * n * m <= MAX_PARTIALS;
    }

    /** k-splits per band: as few as leave the pool {@link #ITEMS_PER_WORKER} items per worker. */
    private static int splits(int m, int k, int width) {
        int bands = Tile.ceilDiv(m, MR), blocks = Tile.ceilDiv(k, KC);
        return Math.clamp(Tile.ceilDiv(ITEMS_PER_WORKER * width, bands), 1, blocks);
    }

    static void run(
            Parallel parallel,
            Scratch scratch,
            Weight w,
            MemorySegment act,
            long aOff,
            int lda,
            MemorySegment r,
            long rOff,
            int ldr,
            int m,
            int n,
            int k) {
        int bands = Tile.ceilDiv(m, MR),
                blocks = Tile.ceilDiv(k, KC),
                splits = splits(m, k, parallel.width());
        int perSplit = Tile.ceilDiv(blocks, splits);
        splits = Tile.ceilDiv(blocks, perSplit); // the last split may be short, never empty
        int items = bands * splits;
        if (scratch.activation.length < n || scratch.activation[0].length < k + Tile.KU) {
            scratch.activation = new float[n][];
            for (int i = 0; i < n; i++) scratch.activation[i] = new float[k + Tile.KU];
        }
        float[][] a = scratch.activation;
        for (int i = 0; i < n; i++) { // the tail past k meets zero panel rows: keep it finite
            MemorySegment.copy(act, JAVA_FLOAT_UNALIGNED, aOff + (long) i * lda * 4, a[i], 0, k);
            Arrays.fill(a[i], k, k + Tile.KU, 0f);
        }
        if (scratch.partials.length < splits * n * m) scratch.partials = new float[splits * n * m];
        float[] partial = scratch.partials;
        int nSplits = splits;

        // two regions: the band x split items into their partials, then one reduce per token
        parallel.forLoop(
                items,
                (item, slot) -> {
                    int band = item / nSplits, split = item - band * nSplits;
                    int l0 = split * perSplit * KC, l1 = Math.min(k, l0 + perSplit * KC);
                    band(
                            w,
                            band * MR,
                            Math.min(MR, m - band * MR),
                            l0,
                            l1,
                            a,
                            n,
                            partial,
                            split * n,
                            m,
                            scratch.slot(slot));
                });
        parallel.forLoop(
                n,
                (t, slot) -> reduce(partial, nSplits, n, m, t, r, rOff, ldr, scratch.slot(slot)));
    }

    /**
     * One item: rows {@code i0 .. i0+rows} over {@code l0 .. l1}, k-block by k-block, for every
     * token, into the partial rows starting at {@code partial[(p0 + t) * m + i0]}.
     */
    private static void band(
            Weight w,
            int i0,
            int rows,
            int l0,
            int l1,
            float[][] a,
            int n,
            float[] partial,
            int p0,
            int m,
            Scratch.Slot s) {
        Rows c = s.c.fit(n + Tile.TR - 1, MR), wT = s.wT.fit(KC + Tile.KU, MR);
        float[] tmp = s.row(KC);
        int lanes = Tile.lanes(rows); // a short last band still sweeps wide
        c.zero(0, n, lanes);
        for (int lb = l0; lb < l1; lb += KC) {
            int len = Math.min(KC, l1 - lb), lenPad = Tile.padK(len);
            for (int i = 0; i < rows; i++) {
                s.decode.row(w, i0 + i, lb, len, len, tmp, 0);
                for (int l = 0; l < len; l++) wT.row(l)[i] = tmp[l];
            }
            wT.zero(len, lenPad, lanes);
            for (int t = 0; t < n; t += Tile.TR) {
                for (int j = 0; j < Tile.TR; j++) {
                    if (t + j < n) System.arraycopy(a[t + j], lb, s.x, j * Tile.XS, lenPad);
                    else Arrays.fill(s.x, j * Tile.XS, j * Tile.XS + lenPad, 0f);
                }
                Tile.sweep(wT.rows(), lenPad, s.x, c.row(t), c.row(t + 1), c.row(t + 2), lanes);
            }
        }
        for (int t = 0; t < n; t++) System.arraycopy(c.row(t), 0, partial, (p0 + t) * m + i0, rows);
    }

    /** Sums token {@code t}'s split partials into its result row. */
    private static void reduce(
            float[] partial,
            int splits,
            int n,
            int m,
            int t,
            MemorySegment r,
            long rOff,
            int ldr,
            Scratch.Slot s) {
        float[] sum = s.acc(m);
        System.arraycopy(partial, t * m, sum, 0, m);
        for (int split = 1; split < splits; split++) {
            int from = (split * n + t) * m;
            for (int i = 0; i < m; i++) sum[i] += partial[from + i];
        }
        MemorySegment.copy(sum, 0, r, JAVA_FLOAT_UNALIGNED, rOff + (long) t * ldr * 4, m);
    }
}
