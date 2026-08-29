package com.qxotic.jam.vector;

import static com.qxotic.jam.vector.VectorSupport.F_LEN;
import static com.qxotic.jam.vector.VectorSupport.F_SPECIES;
import static com.qxotic.jam.vector.VectorSupport.GLOBAL;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static jdk.incubator.vector.VectorOperators.ADD;

import com.oracle.svm.shared.AlwaysInline;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.concurrent.atomic.AtomicInteger;
import jdk.incubator.vector.FloatVector;

/**
 * The dequant-to-scratch F32 gemm behind every Vector JAM dtype (Q8_0, Q4_0, Q1_0, MXFP4, NVFP4 and
 * the k-quants). A dtype kernel supplies one thing, a {@link RowDequant} that decodes a run of one
 * weight row into F32; this class owns the blocking, the packing, the threading and the
 * register-tile sweeps.
 *
 * <p><b>Layouts.</b> Both operands are packed into 64-byte-chunk interleaved layouts so a sweep
 * walks two pointers with constant displacements: an activation tile is {@code
 * [k/F_LEN][NR][F_LEN]} (NR token columns), a weight band is {@code [k/F_LEN][MR][F_LEN]} (MR
 * rows). One MR x NR sweep holds MR*NR accumulators whose lanes run along k and reduces each once
 * at the end.
 *
 * <p><b>Blocking.</b> The sweep is FMA-bound only when the tile it re-reads is L1-resident and the
 * bands stream from L2 (measured on Zen 5: both from L2 gives 240 of ~350 GF/s per core, from L3
 * 160-200). So a task dequantizes a <i>panel</i> of {@link #PANEL_BYTES} of bands into its own L2
 * and sweeps every tile through it while each tile is hot in L1; long k is cut into {@link #KC}
 * blocks (the tile must fit L1 next to a streaming band), every block accumulating into the output,
 * which is zeroed up front so the sweeps never branch on "first block".
 *
 * <p><b>Scheduling.</b> One parallel region per gemm. Tasks are dispensed in order from an atomic
 * counter: first the packing of the activation tiles, then full panels, then a tail of quarter
 * panels so that a late worker ends at most a quarter of a task after the others (the recursive
 * range split handed the small tail out FIRST to stealing workers; measured worker efficiency 82 ->
 * 95% at 16 threads). Small m splits the tokens across tasks too, each task decoding its panel
 * privately - no barriers, no shared decode buffers. Neighbouring panels never share an output
 * cache line ({@link #PANEL_GRANULE}).
 *
 * <p><b>Tiles.</b> 4x4 on C2 (24 live vectors, spill-free on 32 ZMM) and 3x3 on a jvmci JIT (Graal
 * allocates zmm0-15 only). Graal also partially unrolls the sweep loop 2x, which spills the
 * accumulators; the 3x3 sweep therefore advances its pointers by an opaque runtime stride (190 ->
 * 303 GF/s per core, no JVM flag). Both sweeps are branch-free and call-free: a FloatVector
 * crossing a call C2 leaves out of line is boxed, and every ZMM is caller-saved. Override the shape
 * with {@code -Djam.vector.band=3x3|4x4}.
 */
final class BandGemm {

    private BandGemm() {}

    /** Register-tile shape: {@code 3x3} or {@code 4x4}. */
    static final String BAND =
            VectorSupport.jamProp(
                    "jam.vector.band",
                    VectorSupport.WIDE_TILE && VectorSupport.IS_512 ? "4x4" : "3x3");

    static final int MR = BAND.equals("3x3") ? 3 : 4;
    static final int NR = BAND.equals("3x3") ? 3 : 4;

    /** The output element layout. */
    private static final ValueLayout.OfFloat F32 = JAVA_FLOAT_UNALIGNED;

    /** Bytes of one F32 vector, and of one k-chunk of a band / a tile. */
    static final int VB = F_LEN * Float.BYTES;

    static final int BAND_CHUNK = MR * VB;
    static final int TILE_CHUNK = NR * VB;

    /**
     * The chunk strides as non-final statics, opaque to the JIT: the 3x3 sweep's induction stride
     * is then not a compile-time constant and Graal does not partially unroll the loop (see class
     * doc). Never written after class init.
     */
    private static int bandStep = BAND_CHUNK;

    private static int tileStep = TILE_CHUNK;

    /**
     * k-block length in elements ({@code -Djam.vector.kc}, default 512): a tile of NR columns x KC
     * floats plus one streaming band must stay L1-resident - 16 KB at 4x4, which fits any 32 KB
     * L1D. A multiple of 256 so every k-quant super-block stays whole. (Zen 5, 48 KB L1D: 1024
     * measures within run noise of it.)
     */
    static final int KC = Math.max(256, VectorSupport.jamPropInt("jam.vector.kc", 512) / 256 * 256);

    /**
     * Dequant panel cap in bytes ({@code -Djam.vector.panelKb}, default 256 KiB): a panel is what a
     * task sweeps every tile through, so it must stay in the L2 available to its core - half of a
     * 512 KB private L2, or 8 cores x 256 KB of a shared cluster L2. (Zen 5, 1 MB L2: 512 KB
     * measures within run noise of it.)
     */
    static final int PANEL_BYTES = VectorSupport.jamPropInt("jam.vector.panelKb", 256) * 1024;

    /**
     * Panel row granule: a multiple of MR (whole bands) and of 16 rows (64 output bytes), so panels
     * running on different cores never accumulate into a shared output cache line (measured: 3x3
     * panels of 126 rows halved 16-thread throughput through false sharing).
     */
    static final int PANEL_GRANULE = MR * 16 / gcd(MR, 16);

    /** Tasks per worker the plan aims for, before the tail is added. */
    private static final int TASKS_PER_WORKER = 4;

    /**
     * Decode {@code count} elements of one weight row from element offset {@code rowElemOffset}
     * into F32 at byte offset {@code dstBase} of {@code dst} - the only part that differs between
     * dtypes. {@code dst} is routed through {@link VectorSupport#vectorSegment}; the run always
     * starts and ends on a quant-block boundary of the dtype.
     */
    @FunctionalInterface
    interface RowDequant {
        void dequantize(
                MemorySegment w, long rowElemOffset, int count, MemorySegment dst, long dstBase);
    }

    /**
     * {@code o[s*oStride + row] = sum_k w[row][k] * a[s*aStride + k]} for {@code n} token columns
     * and {@code m} weight rows; {@code a} and {@code o} are routed segments addressed from {@code
     * aBase}/{@code oBase}, {@code w} is the raw weight segment read from element {@code wOff}.
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
        gemm(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, scratch, deq, KC);
    }

    /** {@link #gemm} with an explicit k-block cap (tests exercise the blocking with small kc). */
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
            int kcMax) {
        if (n <= 0 || m <= 0) return;
        Plan plan = new Plan(n, m, k, Math.min(kcMax, k));
        MemorySegment packedA = scratch.acquire((long) plan.tiles * NR * k);
        try {
            new Gemm(w, wOff, a, aBase, aStride, o, oBase, oStride, deq, plan, scratch, packedA)
                    .run();
        } finally {
            scratch.release(packedA);
        }
    }

    /** The blocking of one gemm: k-blocks, panels, token splits and the task numbering. */
    private static final class Plan {
        final int n, m, k, kc;
        final int kBlocks, tiles;
        final int panel, tailPanel, bulkRows, bulkPanels, panels;
        final int splits, tilesPerSplit;
        final int packItems, tasks;

        Plan(int n, int m, int k, int kc) {
            this.n = n;
            this.m = m;
            this.k = k;
            this.kc = kc;
            kBlocks = ceilDiv(k, kc);
            tiles = ceilDiv(n, NR);
            panel = panelRows(kc, m);
            // Guided tail: the last ~2 tasks per worker are quarter panels (at least a granule).
            tailPanel = Math.max(PANEL_GRANULE, panel / 4 / PANEL_GRANULE * PANEL_GRANULE);
            int tailRows =
                    tailPanel == panel ? 0 : Math.min(m, 2 * VectorSupport.width() * tailPanel);
            bulkRows = m - tailRows;
            bulkPanels = ceilDiv(bulkRows, panel);
            panels = bulkPanels + ceilDiv(tailRows, tailPanel);
            // Few panels even at one granule (small m): split the tokens across tasks too.
            int wanted = TASKS_PER_WORKER * VectorSupport.width();
            splits = panels >= wanted ? 1 : Math.min(tiles, ceilDiv(wanted, panels));
            tilesPerSplit = ceilDiv(tiles, splits);
            packItems = tiles * kBlocks;
            tasks = packItems + panels * splits;
        }

        /** First row of panel {@code p}. */
        int start(int p) {
            return p < bulkPanels ? p * panel : bulkRows + (p - bulkPanels) * tailPanel;
        }

        /** Rows of panel {@code p}. */
        int rows(int p) {
            return p < bulkPanels
                    ? Math.min(panel, bulkRows - start(p))
                    : Math.min(tailPanel, m - start(p));
        }
    }

    /**
     * Rows per panel for a k-block of {@code kc} and {@code m} rows: a granule multiple within
     * {@link #PANEL_BYTES}, shrunk so that {@code m} yields at least {@link #TASKS_PER_WORKER}
     * panels per worker (never below one granule).
     */
    static int panelRows(int kc, int m) {
        int byCache = PANEL_BYTES / (kc * Float.BYTES);
        int byBalance = m / (TASKS_PER_WORKER * VectorSupport.width());
        int rows = Math.min(byCache, byBalance) / PANEL_GRANULE * PANEL_GRANULE;
        return Math.max(PANEL_GRANULE, rows);
    }

    /** One gemm call: the operands, its plan, and the task bodies. */
    private static final class Gemm {
        final MemorySegment w, a, o, pa;
        final long wOff, aBase, oBase, pab;
        final int aStride, oStride;
        final RowDequant deq;
        final Plan plan;
        final Scratch scratch;
        final AtomicInteger next = new AtomicInteger(), packed = new AtomicInteger();

        Gemm(
                MemorySegment w,
                long wOff,
                MemorySegment a,
                long aBase,
                int aStride,
                MemorySegment o,
                long oBase,
                int oStride,
                RowDequant deq,
                Plan plan,
                Scratch scratch,
                MemorySegment packedA) {
            this.w = w;
            this.wOff = wOff;
            this.a = a;
            this.aBase = aBase;
            this.aStride = aStride;
            this.o = o;
            this.oBase = oBase;
            this.oStride = oStride;
            this.deq = deq;
            this.plan = plan;
            this.scratch = scratch;
            this.pa = VectorSupport.vectorSegment(packedA);
            this.pab = VectorSupport.vectorBase(packedA);
        }

        /** One region: every worker pulls tasks from the counter until none are left. */
        void run() {
            VectorSupport.parallelForEach(
                    0,
                    Math.min(plan.tasks, VectorSupport.width()),
                    worker -> {
                        for (int i; (i = next.getAndIncrement()) < plan.tasks; ) {
                            if (i < plan.packItems) pack(i);
                            else sweep(i - plan.packItems, worker);
                        }
                    });
        }

        /**
         * Pack item {@code i}: tile {@code i % tiles} of k-block {@code i / tiles}. Packed A is
         * k-block-major, so a k-block's tiles are one contiguous stream (prefetcher-friendly).
         */
        void pack(int i) {
            int kb = i / plan.tiles, t = i % plan.tiles;
            int kOff = kb * plan.kc, kcb = Math.min(plan.kc, plan.k - kOff);
            packTile(
                    a,
                    aBase + (long) kOff * 4L,
                    aStride,
                    pa,
                    pab + ((long) kOff * plan.tiles + (long) t * kcb) * NR * 4L,
                    t * NR,
                    plan.n,
                    kcb);
            packed.incrementAndGet();
        }

        /** Sweep task {@code j}: panel {@code j / splits} against tile range {@code j % splits}. */
        void sweep(int j, int slot) {
            int p = j / plan.splits, sp = j % plan.splits;
            int r0 = plan.start(p), rows = plan.rows(p);
            int tLo = sp * plan.tilesPerSplit;
            int tHi = Math.min(plan.tiles, tLo + plan.tilesPerSplit);
            if (tLo >= tHi) return;
            while (packed.get() < plan.packItems) Thread.onSpinWait(); // the pack items go first
            long panelFloats = (long) plan.panel * plan.kc;
            MemorySegment raw = scratch.acquireLocal(slot, panelFloats + (long) MR * plan.kc);
            MemorySegment sv = VectorSupport.vectorSegment(raw);
            long sb = VectorSupport.vectorBase(raw); // the interleaved bands
            long lin = sb + panelFloats * 4L; // MR linear rows: dequant staging
            zeroRows(tLo * NR, Math.min(plan.n, tHi * NR), r0, rows);
            for (int kOff = 0; kOff < plan.k; kOff += plan.kc) {
                int kcb = Math.min(plan.kc, plan.k - kOff);
                dequantPanel(kOff, kcb, sv, sb, lin, r0, rows);
                sweepTiles(kOff, kcb, sv, sb, r0, rows, tLo, tHi);
            }
        }

        /** Zero rows {@code [r0, r0+rows)} of token columns {@code [sLo, sHi)}. */
        void zeroRows(int sLo, int sHi, int r0, int rows) {
            for (int s = sLo; s < sHi; s++)
                o.asSlice(oBase + ((long) s * oStride + r0) * 4L, (long) rows * 4L).fill((byte) 0);
        }

        /**
         * Dequantize k-block {@code [kOff, kOff+kcb)} of rows {@code [r0, r0+rows)} into
         * interleaved bands at {@code sb} (a trailing partial band is zero-padded), staging MR
         * linear rows at {@code lin}.
         */
        void dequantPanel(
                int kOff, int kcb, MemorySegment sv, long sb, long lin, int r0, int rows) {
            long bandBytes = (long) kcb * MR * 4L;
            for (int b = 0; b < ceilDiv(rows, MR); b++) {
                int row0 = r0 + b * MR;
                for (int i = 0; i < MR; i++) {
                    long dst = lin + (long) i * kcb * 4L;
                    if (row0 + i < plan.m)
                        deq.dequantize(w, wOff + (long) (row0 + i) * plan.k + kOff, kcb, sv, dst);
                    else sv.asSlice(dst, (long) kcb * 4L).fill((byte) 0);
                }
                interleave(sv, lin, sb + b * bandBytes, kcb);
            }
        }

        /**
         * Sweep tiles {@code [tLo, tHi)} of k-block {@code kOff} against the bands of rows {@code
         * [r0, r0+rows)}: full tiles x full bands through the branch-free panel sweep, the trailing
         * partial tile / partial band through the edge sweep.
         */
        void sweepTiles(
                int kOff, int kcb, MemorySegment sv, long sb, int r0, int rows, int tLo, int tHi) {
            int bands = ceilDiv(rows, MR), fullBands = rows / MR;
            int fullTiles = Math.min(tHi, plan.n / NR);
            long bandBytes = (long) kcb * MR * 4L, tileBytes = (long) kcb * NR * 4L;
            long kbBase = pab + (long) kOff * plan.tiles * NR * 4L; // this k-block's tiles
            long oStrideBytes = (long) oStride * 4L;
            boolean fast = GLOBAL != null && o == GLOBAL; // absolute output addresses
            if (fast && fullTiles > tLo && fullBands > 0) {
                long out0 = oBase + ((long) tLo * NR * oStride + r0) * 4L;
                long tile0 = kbBase + tLo * tileBytes;
                int tiles = fullTiles - tLo;
                if (MR == 4)
                    sweepPanel44(
                            sv,
                            sb,
                            bandBytes,
                            fullBands,
                            pa,
                            tile0,
                            tileBytes,
                            tiles,
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
                            tile0,
                            tileBytes,
                            tiles,
                            kcb,
                            out0,
                            oStrideBytes);
            }
            for (int t = tLo; t < tHi; t++) {
                int s0 = t * NR, cols = Math.min(NR, plan.n - s0);
                long tile = kbBase + (long) t * tileBytes;
                for (int b = fast && cols == NR ? fullBands : 0; b < bands; b++) {
                    int row0 = r0 + b * MR, rowsValid = Math.min(MR, r0 + rows - row0);
                    if (MR == 4)
                        sweepEdge44(
                                sv,
                                sb + b * bandBytes,
                                pa,
                                tile,
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
                                tile,
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
    }

    /**
     * Pack {@code kc} elements of NR activation columns from {@code s0} (column 0's first element
     * at routed address {@code aBase}) into interleaved {@link #TILE_CHUNK} chunks at {@code
     * dstBase}; columns beyond {@code n} are zero-filled (swept, never stored).
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
        long d = dstBase;
        for (int kk = 0; kk < kc; kk += F_LEN, d += TILE_CHUNK) {
            for (int c = 0; c < NR; c++) {
                FloatVector v =
                        s0 + c < n
                                ? load(a, aBase + ((long) (s0 + c) * aStride + kk) * 4L)
                                : FloatVector.zero(F_SPECIES);
                v.intoMemorySegment(pa, d + (long) c * VB, ByteOrder.LITTLE_ENDIAN);
            }
        }
    }

    /**
     * Interleave MR linear rows of {@code kc} floats at {@code srcBase} into {@link #BAND_CHUNK}
     * chunks at {@code dstBase}.
     */
    static void interleave(MemorySegment sv, long srcBase, long dstBase, int kc) {
        long rowBytes = (long) kc * 4;
        long d = dstBase;
        for (int kk = 0; kk < kc; kk += F_LEN, d += BAND_CHUNK) {
            long kb = (long) kk * 4;
            for (int r = 0; r < MR; r++)
                load(sv, srcBase + r * rowBytes + kb)
                        .intoMemorySegment(sv, d + (long) r * VB, ByteOrder.LITTLE_ENDIAN);
        }
    }

    /**
     * The 3x3 sweep of {@code tiles} full tiles x {@code bands} full bands of one k-block,
     * accumulating into the output at absolute addresses ({@code out0} = o[s0][row0], {@code
     * oStrideBytes} between token columns). Branch-free, call-free; the pointer strides are the
     * opaque statics (no partial unroll on a jvmci JIT).
     */
    static void sweepPanel33(
            MemorySegment w,
            long panel,
            long bandBytes,
            int bands,
            MemorySegment a,
            long tile0,
            long tileBytes,
            int tiles,
            int kc,
            long out0,
            long oStrideBytes) {
        final int wStep = bandStep, aStep = tileStep;
        final long kBytes = (long) kc * MR * 4;
        final MemorySegment g = GLOBAL;
        for (int t = 0; t < tiles; t++) {
            final long tile = tile0 + t * tileBytes;
            long out = out0 + t * NR * oStrideBytes;
            for (int b = 0; b < bands; b++, out += MR * 4L) {
                FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = c00, c02 = c00;
                FloatVector c10 = c00, c11 = c00, c12 = c00;
                FloatVector c20 = c00, c21 = c00, c22 = c00;
                long wp = panel + b * bandBytes, ap = tile;
                for (long end = wp + kBytes; wp < end; wp += wStep, ap += aStep) {
                    FloatVector v0 = load(w, wp), v1 = load(w, wp + VB), v2 = load(w, wp + 2L * VB);
                    FloatVector x0 = load(a, ap), x1 = load(a, ap + VB), x2 = load(a, ap + 2L * VB);
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
                g.set(F32, out, g.get(F32, out) + c00.reduceLanes(ADD));
                g.set(F32, out + 4, g.get(F32, out + 4) + c10.reduceLanes(ADD));
                g.set(F32, out + 8, g.get(F32, out + 8) + c20.reduceLanes(ADD));
                g.set(F32, o1, g.get(F32, o1) + c01.reduceLanes(ADD));
                g.set(F32, o1 + 4, g.get(F32, o1 + 4) + c11.reduceLanes(ADD));
                g.set(F32, o1 + 8, g.get(F32, o1 + 8) + c21.reduceLanes(ADD));
                g.set(F32, o2, g.get(F32, o2) + c02.reduceLanes(ADD));
                g.set(F32, o2 + 4, g.get(F32, o2 + 4) + c12.reduceLanes(ADD));
                g.set(F32, o2 + 8, g.get(F32, o2 + 8) + c22.reduceLanes(ADD));
            }
        }
    }

    /** The 4x4 sweep (C2: 24 live vectors, spill-free on 32 ZMM); see {@link #sweepPanel33}. */
    static void sweepPanel44(
            MemorySegment w,
            long panel,
            long bandBytes,
            int bands,
            MemorySegment a,
            long tile0,
            long tileBytes,
            int tiles,
            int kc,
            long out0,
            long oStrideBytes) {
        final long kBytes = (long) kc * MR * 4;
        final MemorySegment g = GLOBAL;
        for (int t = 0; t < tiles; t++) {
            final long tile = tile0 + t * tileBytes;
            long out = out0 + t * NR * oStrideBytes;
            for (int b = 0; b < bands; b++, out += MR * 4L) {
                FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = c00, c02 = c00, c03 = c00;
                FloatVector c10 = c00, c11 = c00, c12 = c00, c13 = c00;
                FloatVector c20 = c00, c21 = c00, c22 = c00, c23 = c00;
                FloatVector c30 = c00, c31 = c00, c32 = c00, c33 = c00;
                long wp = panel + b * bandBytes, ap = tile;
                for (long end = wp + kBytes; wp < end; wp += BAND_CHUNK, ap += TILE_CHUNK) {
                    FloatVector x0 = load(a, ap), x1 = load(a, ap + VB);
                    FloatVector x2 = load(a, ap + 2L * VB), x3 = load(a, ap + 3L * VB);
                    FloatVector v = load(w, wp);
                    c00 = v.fma(x0, c00);
                    c01 = v.fma(x1, c01);
                    c02 = v.fma(x2, c02);
                    c03 = v.fma(x3, c03);
                    v = load(w, wp + VB);
                    c10 = v.fma(x0, c10);
                    c11 = v.fma(x1, c11);
                    c12 = v.fma(x2, c12);
                    c13 = v.fma(x3, c13);
                    v = load(w, wp + 2L * VB);
                    c20 = v.fma(x0, c20);
                    c21 = v.fma(x1, c21);
                    c22 = v.fma(x2, c22);
                    c23 = v.fma(x3, c23);
                    v = load(w, wp + 3L * VB);
                    c30 = v.fma(x0, c30);
                    c31 = v.fma(x1, c31);
                    c32 = v.fma(x2, c32);
                    c33 = v.fma(x3, c33);
                }
                long o1 = out + oStrideBytes, o2 = o1 + oStrideBytes, o3 = o2 + oStrideBytes;
                g.set(F32, out, g.get(F32, out) + c00.reduceLanes(ADD));
                g.set(F32, out + 4, g.get(F32, out + 4) + c10.reduceLanes(ADD));
                g.set(F32, out + 8, g.get(F32, out + 8) + c20.reduceLanes(ADD));
                g.set(F32, out + 12, g.get(F32, out + 12) + c30.reduceLanes(ADD));
                g.set(F32, o1, g.get(F32, o1) + c01.reduceLanes(ADD));
                g.set(F32, o1 + 4, g.get(F32, o1 + 4) + c11.reduceLanes(ADD));
                g.set(F32, o1 + 8, g.get(F32, o1 + 8) + c21.reduceLanes(ADD));
                g.set(F32, o1 + 12, g.get(F32, o1 + 12) + c31.reduceLanes(ADD));
                g.set(F32, o2, g.get(F32, o2) + c02.reduceLanes(ADD));
                g.set(F32, o2 + 4, g.get(F32, o2 + 4) + c12.reduceLanes(ADD));
                g.set(F32, o2 + 8, g.get(F32, o2 + 8) + c22.reduceLanes(ADD));
                g.set(F32, o2 + 12, g.get(F32, o2 + 12) + c32.reduceLanes(ADD));
                g.set(F32, o3, g.get(F32, o3) + c03.reduceLanes(ADD));
                g.set(F32, o3 + 4, g.get(F32, o3 + 4) + c13.reduceLanes(ADD));
                g.set(F32, o3 + 8, g.get(F32, o3 + 8) + c23.reduceLanes(ADD));
                g.set(F32, o3 + 12, g.get(F32, o3 + 12) + c33.reduceLanes(ADD));
            }
        }
    }

    /**
     * The 3x3 edge sweep: one band against one tile, any {@code rowsValid}/{@code cols},
     * segment-relative output - a panel's trailing partial band, the trailing partial tile, and
     * every band when the output is not the pinned GLOBAL segment.
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
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = c00, c02 = c00;
        FloatVector c10 = c00, c11 = c00, c12 = c00;
        FloatVector c20 = c00, c21 = c00, c22 = c00;
        final int wStep = bandStep, aStep = tileStep;
        for (long end = wp + (long) kc * MR * 4; wp < end; wp += wStep, ap += aStep) {
            FloatVector v0 = load(w, wp), v1 = load(w, wp + VB), v2 = load(w, wp + 2L * VB);
            FloatVector x0 = load(a, ap), x1 = load(a, ap + VB), x2 = load(a, ap + 2L * VB);
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
        long o0 = (long) s0 * oStride + row0, o1 = o0 + oStride, o2 = o1 + oStride;
        add(o, oBase, o0, c00.reduceLanes(ADD));
        if (rowsValid > 1) add(o, oBase, o0 + 1, c10.reduceLanes(ADD));
        if (rowsValid > 2) add(o, oBase, o0 + 2, c20.reduceLanes(ADD));
        if (cols > 1) {
            add(o, oBase, o1, c01.reduceLanes(ADD));
            if (rowsValid > 1) add(o, oBase, o1 + 1, c11.reduceLanes(ADD));
            if (rowsValid > 2) add(o, oBase, o1 + 2, c21.reduceLanes(ADD));
        }
        if (cols > 2) {
            add(o, oBase, o2, c02.reduceLanes(ADD));
            if (rowsValid > 1) add(o, oBase, o2 + 1, c12.reduceLanes(ADD));
            if (rowsValid > 2) add(o, oBase, o2 + 2, c22.reduceLanes(ADD));
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
        FloatVector c00 = FloatVector.zero(F_SPECIES), c01 = c00, c02 = c00, c03 = c00;
        FloatVector c10 = c00, c11 = c00, c12 = c00, c13 = c00;
        FloatVector c20 = c00, c21 = c00, c22 = c00, c23 = c00;
        FloatVector c30 = c00, c31 = c00, c32 = c00, c33 = c00;
        for (long end = wp + (long) kc * MR * 4; wp < end; wp += BAND_CHUNK, ap += TILE_CHUNK) {
            FloatVector x0 = load(a, ap), x1 = load(a, ap + VB);
            FloatVector x2 = load(a, ap + 2L * VB), x3 = load(a, ap + 3L * VB);
            FloatVector v = load(w, wp);
            c00 = v.fma(x0, c00);
            c01 = v.fma(x1, c01);
            c02 = v.fma(x2, c02);
            c03 = v.fma(x3, c03);
            v = load(w, wp + VB);
            c10 = v.fma(x0, c10);
            c11 = v.fma(x1, c11);
            c12 = v.fma(x2, c12);
            c13 = v.fma(x3, c13);
            v = load(w, wp + 2L * VB);
            c20 = v.fma(x0, c20);
            c21 = v.fma(x1, c21);
            c22 = v.fma(x2, c22);
            c23 = v.fma(x3, c23);
            v = load(w, wp + 3L * VB);
            c30 = v.fma(x0, c30);
            c31 = v.fma(x1, c31);
            c32 = v.fma(x2, c32);
            c33 = v.fma(x3, c33);
        }
        long o0 = (long) s0 * oStride + row0, o1 = o0 + oStride;
        long o2 = o1 + oStride, o3 = o2 + oStride;
        add(o, oBase, o0, c00.reduceLanes(ADD));
        if (rowsValid > 1) add(o, oBase, o0 + 1, c10.reduceLanes(ADD));
        if (rowsValid > 2) add(o, oBase, o0 + 2, c20.reduceLanes(ADD));
        if (rowsValid > 3) add(o, oBase, o0 + 3, c30.reduceLanes(ADD));
        if (cols > 1) {
            add(o, oBase, o1, c01.reduceLanes(ADD));
            if (rowsValid > 1) add(o, oBase, o1 + 1, c11.reduceLanes(ADD));
            if (rowsValid > 2) add(o, oBase, o1 + 2, c21.reduceLanes(ADD));
            if (rowsValid > 3) add(o, oBase, o1 + 3, c31.reduceLanes(ADD));
        }
        if (cols > 2) {
            add(o, oBase, o2, c02.reduceLanes(ADD));
            if (rowsValid > 1) add(o, oBase, o2 + 1, c12.reduceLanes(ADD));
            if (rowsValid > 2) add(o, oBase, o2 + 2, c22.reduceLanes(ADD));
            if (rowsValid > 3) add(o, oBase, o2 + 3, c32.reduceLanes(ADD));
        }
        if (cols > 3) {
            add(o, oBase, o3, c03.reduceLanes(ADD));
            if (rowsValid > 1) add(o, oBase, o3 + 1, c13.reduceLanes(ADD));
            if (rowsValid > 2) add(o, oBase, o3 + 2, c23.reduceLanes(ADD));
            if (rowsValid > 3) add(o, oBase, o3 + 3, c33.reduceLanes(ADD));
        }
    }

    /** Add {@code v} into output element {@code elem} (the task's rows were zeroed up front). */
    private static void add(MemorySegment o, long oBase, long elem, float v) {
        long off = oBase + elem * 4;
        o.set(F32, off, o.get(F32, off) + v);
    }

    /** F32 vector load at an absolute byte offset of a routed segment (checks fold). */
    @AlwaysInline("hot Vector API helper: a FloatVector must not cross a call")
    private static FloatVector load(MemorySegment seg, long byteOff) {
        return FloatVector.fromMemorySegment(F_SPECIES, seg, byteOff, ByteOrder.LITTLE_ENDIAN);
    }

    private static int ceilDiv(int a, int b) {
        return (a + b - 1) / b;
    }

    private static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }
}
