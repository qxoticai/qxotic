package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Shared decode-free F32 band gemm - the back half of every dequant-to-scratch kernel (Q8_0, Q4_0,
 * Q1_0, MXFP4, NVFP4 and the k-quants Q4_K/Q5_K/Q6_K). The dtype kernel supplies one thing, a
 * {@link RowDequant} that decodes a run of one weight row into F32; this class owns the blocking,
 * the packing, the threading and the register-tile sweep.
 *
 * <p>Both operands are packed into 64-byte-chunk interleaved layouts so the sweep walks two
 * pointers with constant displacements: an activation tile is {@code [k/F_LEN][NR][F_LEN]} (NR
 * token columns), a weight band is {@code [k/F_LEN][MR][F_LEN]} (MR rows). One MR x NR sweep holds
 * MR*NR F32 accumulators whose lanes run along k and reduces them once at the end.
 *
 * <p>Blocking (measured on Zen 5, 512-bit; the numbers are single-core GF/s of a ~350 peak): the
 * sweep is FMA-bound only when the tile it re-reads is L1-resident and the bands stream from L2 -
 * both streaming from L2 gives 240, from L3 160-200. So the loop order is tile-outer, band-inner: a
 * panel of {@link #PANEL_BYTES} dequantized bands sits in L2, each activation tile is fetched once
 * per panel (L3 traffic of one packed A per panel) and swept against every band while it is hot in
 * L1. Long k is cut into {@link #KC} blocks (the tile must fit L1 next to a streaming band), each
 * block accumulating into the output.
 *
 * <p>Tile shape: 4x4 on C2 (24 live vectors, spill-free on 32 ZMM: 323-330 GF/s) and 3x3 on a jvmci
 * JIT (Graal allocates only zmm0-15: 4x4 spills to 145). Graal additionally partially unrolls the
 * counted sweep loop 2x, which doubles the live loaded vectors and spills the accumulators inside
 * the hot loop (190 GF/s); the 3x3 sweep therefore advances its pointers by a runtime
 * (non-constant) stride, which keeps the loop un-unrolled (303 GF/s) with no JVM flag. Override
 * with {@code -Djam.vector.band=3x3|4x4}.
 */
final class BandGemm {

    private BandGemm() {}

    /** Register-tile shape name: {@code 3x3} or {@code 4x4}. */
    static final String BAND =
            VectorSupport.jamProp(
                    "jam.vector.band",
                    VectorSupport.WIDE_TILE && VectorSupport.IS_512 ? "4x4" : "3x3");

    static final int MR = BAND.equals("3x3") ? 3 : 4;
    static final int NR = BAND.equals("3x3") ? 3 : 4;

    /** Bytes of one F32 vector. */
    static final int VB = VectorSupport.F_LEN * Float.BYTES;

    /** Chunk strides of the packed layouts: MR (band) or NR (tile) vectors per k-chunk. */
    static final int BAND_CHUNK = MR * VB;

    static final int TILE_CHUNK = NR * VB;

    /**
     * Runtime copies of the chunk strides for the 3x3 sweep: non-final statics are opaque to the
     * JIT, so the loop's induction stride is not a compile-time constant and Graal does not
     * partially unroll it (see class doc). Never written after class init.
     */
    private static int bandStepRt = BAND_CHUNK;

    private static int tileStepRt = TILE_CHUNK;

    /**
     * k-block length in elements ({@code -Djam.vector.kc}, default 2048): a tile of NR columns x KC
     * floats (32 KB at 4x4) must stay L1-resident while one band (same size) streams past it. Kept
     * a multiple of 256 so every k-quant super-block stays whole.
     */
    static final int KC =
            Math.max(256, VectorSupport.jamPropInt("jam.vector.kc", 1024) / 256 * 256);

    /**
     * Dequant panel cap in bytes ({@code -Djam.vector.panelKb}, default 256 KiB): the bands of one
     * panel are what an activation tile is swept against while L1-hot, and they must stream from L2
     * - measured 4x4, k=2048: 256 KB panel 323 GF/s, 512 KB 297, 1 MB 239.
     */
    static final int PANEL_BYTES = VectorSupport.jamPropInt("jam.vector.panelKb", 512) * 1024;

    /**
     * Decode one weight row run ({@code count} elements from element offset {@code rowElemOffset})
     * into F32 at byte offset {@code dstBase} of {@code dst} - the ONLY part that differs between
     * the dequant-to-scratch dtypes. {@code dst} is pre-routed through {@link
     * VectorSupport#vectorSegment}; {@code dstBase} is the matching routed byte base. The run
     * always starts and ends on a quant-block boundary of the dtype.
     */
    @FunctionalInterface
    interface RowDequant {
        void dequantize(
                MemorySegment w, long rowElemOffset, int count, MemorySegment dst, long dstBase);
    }

    /**
     * Panel row granule: a multiple of MR (whole bands) and of 16 rows (64 output bytes), so
     * neighbouring panels - which run on different cores and accumulate into their rows of every
     * token column - never share an output cache line (measured: 3x3 panels of 126 rows halved
     * 16-thread throughput through false sharing).
     */
    static final int PANEL_GRANULE = MR * 16 / gcd(MR, 16);

    private static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    /** Tasks per worker the driver aims for: the pool balances the tail at task granularity. */
    static final int TASKS_PER_WORKER = VectorSupport.jamPropInt("jam.vector.panelTasks", 4);

    /**
     * Rows per panel for a k-block of {@code kc} and {@code m} rows: a granule multiple within
     * {@link #PANEL_BYTES}, shrunk (down to one granule) so that {@code m} yields at least {@link
     * #TASKS_PER_WORKER} panels per worker - 16 workers on 16 one-panel tasks ran at half the speed
     * of 64 quarter-size panels (any late thread is the whole gemm's tail).
     */
    static int panelRows(int kc, int m) {
        int byCache = PANEL_BYTES / (kc * Float.BYTES);
        int byBalance = m / (TASKS_PER_WORKER * VectorSupport.PARALLELISM);
        int rows = Math.min(byCache, byBalance) / PANEL_GRANULE * PANEL_GRANULE;
        return Math.max(PANEL_GRANULE, rows);
    }

    /** k-block length for a row length {@code k}: whole rows up to {@link #KC}. */
    static int kBlock(int k) {
        return Math.min(k, KC);
    }

    /**
     * The dequant-to-scratch band gemm: {@code o[s*oStride + row] = sum_k w[row][k] * a[s*aStride +
     * k]} for {@code n} token columns and {@code m} weight rows. {@code deq} decodes weight rows;
     * it is called per row and per k-block, so the whole decode is amortized over every column.
     */
    static void gemm(
            MemorySegment w,
            MemorySegment a,
            long aBase,
            MemorySegment o,
            long oBase,
            int aStride,
            int oStride,
            int n,
            int m,
            int k,
            long wOff,
            Scratch scratch,
            RowDequant deq) {
        gemm(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, scratch, deq, kBlock(k));
    }

    /**
     * {@link #gemm} with an explicit k-block length (tests exercise the blocking with small kc).
     */
    static void gemm(
            MemorySegment w,
            MemorySegment a,
            long aBase,
            MemorySegment o,
            long oBase,
            int aStride,
            int oStride,
            int n,
            int m,
            int k,
            long wOff,
            Scratch scratch,
            RowDequant deq,
            int kcRequested) {
        if (n <= 0 || m <= 0) return;
        final int kc = Math.min(kcRequested, k);
        final int tiles = (n + NR - 1) / NR;
        final int kBlocks = (k + kc - 1) / kc;
        final int panel = panelRows(kc, m);
        final int panels = (m + panel - 1) / panel;
        final long panelFloats = (long) panel * kc;
        // Few panels even at one granule (small m): split the tokens too. Each task dequantizes
        // its panel privately (redundant across splits, but a granule panel is cheap) - no
        // barriers, no shared buffers.
        final int wanted = TASKS_PER_WORKER * VectorSupport.PARALLELISM;
        final int splits = panels >= wanted ? 1 : Math.min(tiles, (wanted + panels - 1) / panels);
        final int tilesPerSplit = (tiles + splits - 1) / splits;

        // Packed A, k-block-major: block kb holds its `tiles` tiles back to back, so a panel's
        // tile stream is one contiguous run of tiles*NR*kcb floats (prefetcher-friendly).
        MemorySegment packedA = scratch.acquire((long) tiles * NR * k);
        MemorySegment pa = VectorSupport.vectorSegment(packedA);
        long pab = VectorSupport.vectorBase(packedA);
        try {
            VectorSupport.parallelFor(
                    0,
                    tiles * kBlocks,
                    i -> {
                        int kb = i / tiles, t = i % tiles;
                        int kOff = kb * kc, kcb = Math.min(kc, k - kOff);
                        packTile(
                                a,
                                aBase + (long) kOff * 4L,
                                aStride,
                                pa,
                                pab + ((long) kOff * tiles + (long) t * kcb) * NR * 4L,
                                t * NR,
                                n,
                                kcb);
                    });
            // One (panel, tile range) per task: its dequantized bands are private (L2-resident)
            // and swept against its tiles; the pool balances at task granularity.
            VectorSupport.parallelForEach(
                    0,
                    panels * splits,
                    i -> {
                        int p = i / splits, sp = i % splits;
                        int r0 = p * panel, rows = Math.min(m - r0, panel);
                        int tLo = sp * tilesPerSplit, tHi = Math.min(tiles, tLo + tilesPerSplit);
                        if (tLo >= tHi) return;
                        MemorySegment raw = scratch.acquireLocal(panelFloats + (long) MR * kc);
                        MemorySegment sv = VectorSupport.vectorSegment(raw);
                        long sb = VectorSupport.vectorBase(raw); // interleaved bands
                        long lin = sb + panelFloats * 4L; // MR linear rows, dequant staging
                        zeroRows(o, oBase, oStride, tLo * NR, Math.min(n, tHi * NR), r0, rows);
                        for (int kOff = 0; kOff < k; kOff += kc) {
                            int kcb = Math.min(kc, k - kOff);
                            dequantPanel(w, wOff, m, k, kOff, kcb, sv, sb, lin, r0, rows, deq);
                            sweepTiles(
                                    pa, pab, o, oBase, oStride, n, k, kOff, kcb, tiles, sv, sb, r0,
                                    rows, tLo, tHi);
                        }
                    });
        } finally {
            scratch.release(packedA);
        }
    }

    /**
     * Zero rows {@code [r0, r0+rows)} of token columns {@code [sLo, sHi)}: the sweeps always
     * accumulate into the output (no first-block branch for a JIT to unswitch on). Rows are
     * contiguous per token.
     */
    private static void zeroRows(
            MemorySegment o, long oBase, int oStride, int sLo, int sHi, int r0, int rows) {
        for (int s = sLo; s < sHi; s++)
            o.asSlice(oBase + ((long) s * oStride + r0) * 4L, (long) rows * 4L).fill((byte) 0);
    }

    /**
     * Dequantize k-block {@code [kOff, kOff+kcb)} of rows {@code [r0, r0+rows)} into interleaved
     * bands at {@code sb} (a trailing partial band is zero-padded), staging MR linear rows at
     * {@code lin}.
     */
    private static void dequantPanel(
            MemorySegment w,
            long wOff,
            int m,
            int k,
            int kOff,
            int kcb,
            MemorySegment sv,
            long sb,
            long lin,
            int r0,
            int rows,
            RowDequant deq) {
        final int bands = (rows + MR - 1) / MR;
        final long bandBytes = (long) kcb * MR * 4L;
        for (int b = 0; b < bands; b++) {
            int row0 = r0 + b * MR;
            for (int i = 0; i < MR; i++) {
                long dst = lin + (long) i * kcb * 4L;
                if (row0 + i < m)
                    deq.dequantize(w, wOff + (long) (row0 + i) * k + kOff, kcb, sv, dst);
                else sv.asSlice(dst, (long) kcb * 4L).fill((byte) 0);
            }
            interleave(sv, lin, sb + b * bandBytes, kcb);
        }
    }

    /**
     * Sweep tiles {@code [tLo, tHi)} of k-block {@code kOff} against the bands of rows {@code [r0,
     * r0+rows)} at {@code sb}: full tiles x full bands through the branch-free panel sweep, the
     * trailing partial tile / partial band through the edge sweep.
     */
    private static void sweepTiles(
            MemorySegment pa,
            long pab,
            MemorySegment o,
            long oBase,
            int oStride,
            int n,
            int k,
            int kOff,
            int kcb,
            int tiles,
            MemorySegment sv,
            long sb,
            int r0,
            int rows,
            int tLo,
            int tHi) {
        final int bands = (rows + MR - 1) / MR;
        final int fullBands = rows / MR;
        final int fullTiles = Math.min(tHi, n / NR);
        final long bandBytes = (long) kcb * MR * 4L;
        final long kbBase = pab + (long) kOff * tiles * NR * 4L; // this k-block's tiles
        final long tileBytes = (long) kcb * NR * 4L;
        final long oStrideBytes = (long) oStride * 4L;
        final boolean fast = VectorSupport.GLOBAL != null && o == VectorSupport.GLOBAL;
        if (fast && fullTiles > tLo && fullBands > 0) {
            long out0 = oBase + ((long) tLo * NR * oStride + r0) * 4L; // absolute: o is GLOBAL
            if (MR == 4)
                sweepPanel44(
                        sv,
                        sb,
                        bandBytes,
                        fullBands,
                        pa,
                        kbBase + tLo * tileBytes,
                        tileBytes,
                        fullTiles - tLo,
                        kcb,
                        out0,
                        oStrideBytes);
            else
                sweepPanel33(
                        sv,
                        sb,
                        bandBytes,
                        fullBands,
                        pa,
                        kbBase + tLo * tileBytes,
                        tileBytes,
                        fullTiles - tLo,
                        kcb,
                        out0,
                        oStrideBytes);
        }
        for (int t = tLo; t < tHi; t++) {
            int s0 = t * NR;
            int cols = Math.min(NR, n - s0);
            long tap = kbBase + (long) t * tileBytes;
            int bFrom = fast && cols == NR ? fullBands : 0;
            for (int b = bFrom; b < bands; b++) {
                int row0 = r0 + b * MR;
                int rowsValid = Math.min(MR, r0 + rows - row0);
                if (MR == 4)
                    sweepEdge44(
                            sv,
                            sb + b * bandBytes,
                            pa,
                            tap,
                            kcb,
                            o,
                            oBase,
                            oStride,
                            row0,
                            s0,
                            rowsValid,
                            cols);
                else
                    sweepEdge33(
                            sv,
                            sb + b * bandBytes,
                            pa,
                            tap,
                            kcb,
                            o,
                            oBase,
                            oStride,
                            row0,
                            s0,
                            rowsValid,
                            cols);
            }
        }
    }

    /**
     * Pack {@code kc} elements of NR activation columns starting at {@code s0} (from the routed
     * byte address {@code aBase} of column 0's first element) into interleaved {@link #TILE_CHUNK}
     * chunks at {@code dstBase}; columns beyond {@code n} are zero-filled (swept, never stored).
     */
    static void packTile(
            MemorySegment a,
            long aBase,
            int aStride,
            MemorySegment pa,
            long dstBase,
            int s0,
            int n,
            int kc) {
        int fl = VectorSupport.F_LEN;
        long d = dstBase;
        for (int kk = 0; kk < kc; kk += fl, d += TILE_CHUNK) {
            for (int c = 0; c < NR; c++) {
                FloatVector v =
                        s0 + c < n
                                ? av(a, aBase, (long) (s0 + c) * aStride + kk)
                                : FloatVector.zero(F_SPECIES);
                v.intoMemorySegment(pa, d + (long) c * VB, ByteOrder.LITTLE_ENDIAN);
            }
        }
    }

    /**
     * Interleave MR linear scratch rows of {@code kc} floats (at {@code srcBase}) into {@link
     * #BAND_CHUNK} chunks at {@code dstBase}.
     */
    static void interleave(MemorySegment sv, long srcBase, long dstBase, int kc) {
        int fl = VectorSupport.F_LEN;
        long rowBytes = (long) kc * 4;
        long d = dstBase;
        for (int kk = 0; kk < kc; kk += fl, d += BAND_CHUNK) {
            long kb = (long) kk * 4;
            for (int r = 0; r < MR; r++)
                wv(sv, srcBase + r * rowBytes + kb)
                        .intoMemorySegment(sv, d + (long) r * VB, ByteOrder.LITTLE_ENDIAN);
        }
    }

    /**
     * The 3x3 two-pointer sweep over {@code tiles} full tiles x {@code bands} full bands of one
     * k-block, accumulating into the output at absolute addresses ({@code out0} = address of
     * o[0][row0]; {@code oStrideBytes} between token columns). The whole loop nest lives here so
     * the caller has one cold call per k-block: a jvmci JIT that inlined the per-tile sweep into
     * the panel method produced spill-heavy code (measured 30% slower once it kicked in).
     * Branch-free: no conditionals for a JIT to unswitch on, no calls. The pointer strides are read
     * from opaque statics so a jvmci JIT does not partially unroll the loop (its 16-register
     * allocator would spill the accumulators; see class doc).
     */
    static void sweepPanel33(
            MemorySegment w,
            long panelBase,
            long bandBytes,
            int bands,
            MemorySegment a,
            long tiles0,
            long tileBytes,
            int tiles,
            int kc,
            long out0,
            long oStrideBytes) {
        final int wStep = bandStepRt, aStep = tileStepRt;
        final long kBytes = (long) kc * MR * 4;
        final MemorySegment g = VectorSupport.GLOBAL;
        for (int t = 0; t < tiles; t++) {
            final long tile = tiles0 + t * tileBytes;
            long out = out0 + t * NR * oStrideBytes;
            for (int b = 0; b < bands; b++, out += MR * 4L) {
                FloatVector c00 = FloatVector.zero(F_SPECIES),
                        c01 = FloatVector.zero(F_SPECIES),
                        c02 = FloatVector.zero(F_SPECIES);
                FloatVector c10 = FloatVector.zero(F_SPECIES),
                        c11 = FloatVector.zero(F_SPECIES),
                        c12 = FloatVector.zero(F_SPECIES);
                FloatVector c20 = FloatVector.zero(F_SPECIES),
                        c21 = FloatVector.zero(F_SPECIES),
                        c22 = FloatVector.zero(F_SPECIES);
                long wp = panelBase + b * bandBytes, ap = tile;
                for (long end = wp + kBytes; wp < end; wp += wStep, ap += aStep) {
                    FloatVector v0 = wv(w, wp), v1 = wv(w, wp + VB), v2 = wv(w, wp + 2L * VB);
                    FloatVector x0 = wv(a, ap), x1 = wv(a, ap + VB), x2 = wv(a, ap + 2L * VB);
                    c00 = v0.fma(x0, c00);
                    c01 = v0.fma(x1, c01);
                    c02 = v0.fma(x2, c02);
                    c10 = v1.fma(x0, c10);
                    c11 = v1.fma(x1, c11);
                    c12 = v1.fma(x2, c12);
                    c20 = v2.fma(x0, c20);
                    c21 = v2.fma(x1, c21);
                    c22 = v2.fma(x2, c22);
                }
                long o1 = out + oStrideBytes, o2 = o1 + oStrideBytes;
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out,
                        g.get(JAVA_FLOAT_UNALIGNED, out) + c00.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, out + 4)
                                + c10.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, out + 8)
                                + c20.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1,
                        g.get(JAVA_FLOAT_UNALIGNED, o1) + c01.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1 + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, o1 + 4) + c11.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1 + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, o1 + 8) + c21.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2,
                        g.get(JAVA_FLOAT_UNALIGNED, o2) + c02.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2 + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, o2 + 4) + c12.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2 + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, o2 + 8) + c22.reduceLanes(VectorOperators.ADD));
            }
        }
    }

    /** The 4x4 two-pointer sweep (C2: 24 live vectors, spill-free on 32 ZMM); see 3x3. */
    static void sweepPanel44(
            MemorySegment w,
            long panelBase,
            long bandBytes,
            int bands,
            MemorySegment a,
            long tiles0,
            long tileBytes,
            int tiles,
            int kc,
            long out0,
            long oStrideBytes) {
        final long kBytes = (long) kc * MR * 4;
        final MemorySegment g = VectorSupport.GLOBAL;
        for (int t = 0; t < tiles; t++) {
            final long tile = tiles0 + t * tileBytes;
            long out = out0 + t * NR * oStrideBytes;
            for (int b = 0; b < bands; b++, out += MR * 4L) {
                FloatVector c00 = FloatVector.zero(F_SPECIES),
                        c01 = FloatVector.zero(F_SPECIES),
                        c02 = FloatVector.zero(F_SPECIES),
                        c03 = FloatVector.zero(F_SPECIES);
                FloatVector c10 = FloatVector.zero(F_SPECIES),
                        c11 = FloatVector.zero(F_SPECIES),
                        c12 = FloatVector.zero(F_SPECIES),
                        c13 = FloatVector.zero(F_SPECIES);
                FloatVector c20 = FloatVector.zero(F_SPECIES),
                        c21 = FloatVector.zero(F_SPECIES),
                        c22 = FloatVector.zero(F_SPECIES),
                        c23 = FloatVector.zero(F_SPECIES);
                FloatVector c30 = FloatVector.zero(F_SPECIES),
                        c31 = FloatVector.zero(F_SPECIES),
                        c32 = FloatVector.zero(F_SPECIES),
                        c33 = FloatVector.zero(F_SPECIES);
                long wp = panelBase + b * bandBytes, ap = tile;
                for (long end = wp + kBytes; wp < end; wp += BAND_CHUNK, ap += TILE_CHUNK) {
                    FloatVector x0 = wv(a, ap), x1 = wv(a, ap + VB);
                    FloatVector x2 = wv(a, ap + 2L * VB), x3 = wv(a, ap + 3L * VB);
                    FloatVector v = wv(w, wp);
                    c00 = v.fma(x0, c00);
                    c01 = v.fma(x1, c01);
                    c02 = v.fma(x2, c02);
                    c03 = v.fma(x3, c03);
                    v = wv(w, wp + VB);
                    c10 = v.fma(x0, c10);
                    c11 = v.fma(x1, c11);
                    c12 = v.fma(x2, c12);
                    c13 = v.fma(x3, c13);
                    v = wv(w, wp + 2L * VB);
                    c20 = v.fma(x0, c20);
                    c21 = v.fma(x1, c21);
                    c22 = v.fma(x2, c22);
                    c23 = v.fma(x3, c23);
                    v = wv(w, wp + 3L * VB);
                    c30 = v.fma(x0, c30);
                    c31 = v.fma(x1, c31);
                    c32 = v.fma(x2, c32);
                    c33 = v.fma(x3, c33);
                }
                long o1 = out + oStrideBytes, o2 = o1 + oStrideBytes, o3 = o2 + oStrideBytes;
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out,
                        g.get(JAVA_FLOAT_UNALIGNED, out) + c00.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, out + 4)
                                + c10.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, out + 8)
                                + c20.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        out + 12,
                        g.get(JAVA_FLOAT_UNALIGNED, out + 12)
                                + c30.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1,
                        g.get(JAVA_FLOAT_UNALIGNED, o1) + c01.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1 + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, o1 + 4) + c11.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1 + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, o1 + 8) + c21.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o1 + 12,
                        g.get(JAVA_FLOAT_UNALIGNED, o1 + 12)
                                + c31.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2,
                        g.get(JAVA_FLOAT_UNALIGNED, o2) + c02.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2 + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, o2 + 4) + c12.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2 + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, o2 + 8) + c22.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o2 + 12,
                        g.get(JAVA_FLOAT_UNALIGNED, o2 + 12)
                                + c32.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o3,
                        g.get(JAVA_FLOAT_UNALIGNED, o3) + c03.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o3 + 4,
                        g.get(JAVA_FLOAT_UNALIGNED, o3 + 4) + c13.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o3 + 8,
                        g.get(JAVA_FLOAT_UNALIGNED, o3 + 8) + c23.reduceLanes(VectorOperators.ADD));
                g.set(
                        JAVA_FLOAT_UNALIGNED,
                        o3 + 12,
                        g.get(JAVA_FLOAT_UNALIGNED, o3 + 12)
                                + c33.reduceLanes(VectorOperators.ADD));
            }
        }
    }

    /**
     * The 3x3 edge sweep: one band, any {@code rowsValid}/{@code cols}, segment-relative output.
     * Used for a panel's trailing partial band, the trailing partial tile, and when the output is
     * not the pinned GLOBAL segment.
     */
    static void sweepEdge33(
            MemorySegment w,
            long wp,
            MemorySegment a,
            long ap,
            int kc,
            MemorySegment o,
            long oBase,
            int oStride,
            int row0,
            int s0,
            int rowsValid,
            int cols) {
        FloatVector c00 = FloatVector.zero(F_SPECIES),
                c01 = FloatVector.zero(F_SPECIES),
                c02 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES),
                c11 = FloatVector.zero(F_SPECIES),
                c12 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES),
                c21 = FloatVector.zero(F_SPECIES),
                c22 = FloatVector.zero(F_SPECIES);
        final int wStep = bandStepRt, aStep = tileStepRt;
        for (long end = wp + (long) kc * MR * 4; wp < end; wp += wStep, ap += aStep) {
            FloatVector v0 = wv(w, wp), v1 = wv(w, wp + VB), v2 = wv(w, wp + 2L * VB);
            FloatVector x0 = wv(a, ap), x1 = wv(a, ap + VB), x2 = wv(a, ap + 2L * VB);
            c00 = v0.fma(x0, c00);
            c01 = v0.fma(x1, c01);
            c02 = v0.fma(x2, c02);
            c10 = v1.fma(x0, c10);
            c11 = v1.fma(x1, c11);
            c12 = v1.fma(x2, c12);
            c20 = v2.fma(x0, c20);
            c21 = v2.fma(x1, c21);
            c22 = v2.fma(x2, c22);
        }
        long o0 = (long) s0 * oStride + row0;
        put(o, oBase, o0, c00.reduceLanes(VectorOperators.ADD));
        if (rowsValid > 1) put(o, oBase, o0 + 1, c10.reduceLanes(VectorOperators.ADD));
        if (rowsValid > 2) put(o, oBase, o0 + 2, c20.reduceLanes(VectorOperators.ADD));
        if (cols > 1) {
            long o1 = o0 + oStride;
            put(o, oBase, o1, c01.reduceLanes(VectorOperators.ADD));
            if (rowsValid > 1) put(o, oBase, o1 + 1, c11.reduceLanes(VectorOperators.ADD));
            if (rowsValid > 2) put(o, oBase, o1 + 2, c21.reduceLanes(VectorOperators.ADD));
            if (cols > 2) {
                long o2 = o1 + oStride;
                put(o, oBase, o2, c02.reduceLanes(VectorOperators.ADD));
                if (rowsValid > 1) put(o, oBase, o2 + 1, c12.reduceLanes(VectorOperators.ADD));
                if (rowsValid > 2) put(o, oBase, o2 + 2, c22.reduceLanes(VectorOperators.ADD));
            }
        }
    }

    /** The 4x4 edge sweep; see {@link #sweepEdge33}. */
    static void sweepEdge44(
            MemorySegment w,
            long wp,
            MemorySegment a,
            long ap,
            int kc,
            MemorySegment o,
            long oBase,
            int oStride,
            int row0,
            int s0,
            int rowsValid,
            int cols) {
        FloatVector c00 = FloatVector.zero(F_SPECIES),
                c01 = FloatVector.zero(F_SPECIES),
                c02 = FloatVector.zero(F_SPECIES),
                c03 = FloatVector.zero(F_SPECIES);
        FloatVector c10 = FloatVector.zero(F_SPECIES),
                c11 = FloatVector.zero(F_SPECIES),
                c12 = FloatVector.zero(F_SPECIES),
                c13 = FloatVector.zero(F_SPECIES);
        FloatVector c20 = FloatVector.zero(F_SPECIES),
                c21 = FloatVector.zero(F_SPECIES),
                c22 = FloatVector.zero(F_SPECIES),
                c23 = FloatVector.zero(F_SPECIES);
        FloatVector c30 = FloatVector.zero(F_SPECIES),
                c31 = FloatVector.zero(F_SPECIES),
                c32 = FloatVector.zero(F_SPECIES),
                c33 = FloatVector.zero(F_SPECIES);
        for (long end = wp + (long) kc * MR * 4; wp < end; wp += BAND_CHUNK, ap += TILE_CHUNK) {
            FloatVector x0 = wv(a, ap), x1 = wv(a, ap + VB);
            FloatVector x2 = wv(a, ap + 2L * VB), x3 = wv(a, ap + 3L * VB);
            FloatVector v = wv(w, wp);
            c00 = v.fma(x0, c00);
            c01 = v.fma(x1, c01);
            c02 = v.fma(x2, c02);
            c03 = v.fma(x3, c03);
            v = wv(w, wp + VB);
            c10 = v.fma(x0, c10);
            c11 = v.fma(x1, c11);
            c12 = v.fma(x2, c12);
            c13 = v.fma(x3, c13);
            v = wv(w, wp + 2L * VB);
            c20 = v.fma(x0, c20);
            c21 = v.fma(x1, c21);
            c22 = v.fma(x2, c22);
            c23 = v.fma(x3, c23);
            v = wv(w, wp + 3L * VB);
            c30 = v.fma(x0, c30);
            c31 = v.fma(x1, c31);
            c32 = v.fma(x2, c32);
            c33 = v.fma(x3, c33);
        }
        long o0 = (long) s0 * oStride + row0;
        put(o, oBase, o0, c00.reduceLanes(VectorOperators.ADD));
        if (rowsValid > 1) put(o, oBase, o0 + 1, c10.reduceLanes(VectorOperators.ADD));
        if (rowsValid > 2) put(o, oBase, o0 + 2, c20.reduceLanes(VectorOperators.ADD));
        if (rowsValid > 3) put(o, oBase, o0 + 3, c30.reduceLanes(VectorOperators.ADD));
        if (cols > 1) {
            long o1 = o0 + oStride;
            put(o, oBase, o1, c01.reduceLanes(VectorOperators.ADD));
            if (rowsValid > 1) put(o, oBase, o1 + 1, c11.reduceLanes(VectorOperators.ADD));
            if (rowsValid > 2) put(o, oBase, o1 + 2, c21.reduceLanes(VectorOperators.ADD));
            if (rowsValid > 3) put(o, oBase, o1 + 3, c31.reduceLanes(VectorOperators.ADD));
            if (cols > 2) {
                long o2 = o1 + oStride;
                put(o, oBase, o2, c02.reduceLanes(VectorOperators.ADD));
                if (rowsValid > 1) put(o, oBase, o2 + 1, c12.reduceLanes(VectorOperators.ADD));
                if (rowsValid > 2) put(o, oBase, o2 + 2, c22.reduceLanes(VectorOperators.ADD));
                if (rowsValid > 3) put(o, oBase, o2 + 3, c32.reduceLanes(VectorOperators.ADD));
                if (cols > 3) {
                    long o3 = o2 + oStride;
                    put(o, oBase, o3, c03.reduceLanes(VectorOperators.ADD));
                    if (rowsValid > 1) put(o, oBase, o3 + 1, c13.reduceLanes(VectorOperators.ADD));
                    if (rowsValid > 2) put(o, oBase, o3 + 2, c23.reduceLanes(VectorOperators.ADD));
                    if (rowsValid > 3) put(o, oBase, o3 + 3, c33.reduceLanes(VectorOperators.ADD));
                }
            }
        }
    }

    /** Add {@code v} into output element {@code elem} (the panel's rows were zeroed up front). */
    private static void put(MemorySegment o, long oBase, long elem, float v) {
        long off = oBase + elem * 4;
        o.set(JAVA_FLOAT_UNALIGNED, off, o.get(JAVA_FLOAT_UNALIGNED, off) + v);
    }

    /** Scratch/packed load at absolute byte offset (pinned route: checks fold). */
    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see"
                    + " hotspot_compile_commands)")
    private static FloatVector wv(MemorySegment w, long byteOff) {
        return FloatVector.fromMemorySegment(F_SPECIES, w, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    @AlwaysInline(
            "hot Vector API helper: escaping FloatVector boxes per call (see"
                    + " hotspot_compile_commands)")
    private static FloatVector av(MemorySegment a, long aBase, long elem) {
        return FloatVector.fromMemorySegment(
                F_SPECIES, a, aBase + elem * 4, ByteOrder.LITTLE_ENDIAN);
    }

    /** Scalar F32 store at element index {@code elem} of the output (token-major). */
    static void store(MemorySegment o, long oBase, long elem, float v) {
        o.set(JAVA_FLOAT_UNALIGNED, oBase + elem * 4, v);
    }
}
