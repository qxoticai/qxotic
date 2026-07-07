package com.qxotic.jam;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Cross-backend parity for the JAM matmul backends, driven through the raw {@link JAM} segment contract:
 * {@link ScalarJAM} (the reference floor, jam-scalar), {@link VectorJAM} (jam-vector), and {@link NativeJAM}
 * (jam-native, only when libjam loads). Every backend runs on the same synthetic quantized weights
 * ({@link QuantWeights}, exact-value encoders) and is checked against a double-precision reference over those
 * values. This is the JUnit 6 form of the former {@code JamBackendTest} main() harness, relocated to jam now
 * that the kernels live here.
 *
 * <p>Covers: gemm ({@code n>1}) for every dtype; gemv ({@code n==1}) for the dtype-agnostic backends;
 * VectorJAM's decline contract; cross-backend agreement on identical inputs; and tile-selection sanity.
 */
class JamBackendParityTest {

    private static final long SEED = 0x5A11C0DEL;
    private static final Arena A = Arena.ofAuto();

    private static final JAM SCALAR = new ScalarJAM();
    private static final JAM VECTOR = new VectorJAM();
    /** The native backend if libjam loaded (its class init runs NativeLoader.load), else null -> skipped. */
    private static final JAM NATIVE = tryNative();

    /** Dtypes with a VectorJAM register-tiled / band gemm (the rest decline on VectorJAM). */
    private static final int[] TILEABLE = {JAM.Q8_0, JAM.Q4_0, JAM.Q4_K, JAM.Q5_K, JAM.Q6_K, JAM.MXFP4, JAM.NVFP4};
    /** All JAM weight dtypes (tileable + the dense floats handled only by the dot floor / native). */
    private static final int[] ALL = {JAM.Q8_0, JAM.Q4_0, JAM.Q4_K, JAM.Q5_K, JAM.Q6_K, JAM.MXFP4, JAM.NVFP4,
                                      JAM.F16, JAM.BF16, JAM.F32};

    private static JAM tryNative() {
        try {
            JAM n = NativeJAM.global();
            // a tiny real mm forces the native lib to actually load + run; null it out on any failure
            try (Arena ar = Arena.ofConfined()) {
                MemorySegment w = ar.allocate(16), x = ar.allocate(16), r = ar.allocate(4);
                n.mm(w, 0, JAM.F32, 1, x, 0, JAM.F32, 1, r, 0, JAM.F32, 1, 1, 1, 1);   // m=n=k=1
            }
            return n;
        } catch (Throwable t) {
            return null;
        }
    }

    // ---- one matmul through the contract: contiguous weight [m×k], activation [n×k], result [n×m] ----

    private static int run(JAM backend, QuantWeights.Weight w, MemorySegment a, MemorySegment r,
                           int m, int n, int k) {
        return backend.mm(w.seg(), 0, w.tag(), k, a, 0, JAM.F32, k, r, 0, JAM.F32, m, m, n, k);
    }

    /** Assert {@code backend} computes the reference matmul for dtype {@code tag} at the given shape. */
    private static void parity(JAM backend, String name, int tag, int m, int n, int k, double relTol) {
        Random rng = new Random(SEED ^ tag ^ name.hashCode() ^ ((long) m << 20) ^ ((long) n << 8) ^ k);
        QuantWeights.Weight w = QuantWeights.encode(tag, m, k, A, rng);
        float[] av = QuantWeights.gaussians(n * k, rng);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment r = A.allocate((long) n * m * 4, 64);

        assertEquals(JAM.OK, run(backend, w, a, r, m, n, k), name + " status");
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i += 7) {
            double[] ref = QuantWeights.refDot(w, av, i, j, k);   // [dot, sumAbs]
            float got = r.get(ValueLayout.JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
            assertEquals(ref[0], got, relTol * ref[1] + 1e-3, name + "[token " + j + ", row " + i + "]");
        }
    }

    /** ScalarJAM is full-F32 (tight); the tiled/native paths quantize activations to int8 (error ∝ Σ|w·a|). */
    private static double tol(JAM backend) { return backend == SCALAR ? 1e-3 : 1e-2; }

    // ---- 1. gemm (n>1) for every dtype, on each applicable backend ----

    @Test void scalarGemmEveryDtype() {
        for (int t : ALL) for (int n : new int[]{7, 8, 13, 16}) parity(SCALAR, "ScalarJAM " + name(t), t, 104, n, blockK(t), tol(SCALAR));
    }

    @Test void vectorGemmTileableDtypes() {
        for (int t : TILEABLE) for (int n : new int[]{7, 8, 13, 16}) parity(VECTOR, "VectorJAM " + name(t), t, 104, n, blockK(t), tol(VECTOR));
    }

    @Test void nativeGemmEveryDtype() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int t : ALL) for (int n : new int[]{7, 8, 13, 16}) parity(NATIVE, "NativeJAM " + name(t), t, 104, n, blockK(t), tol(NATIVE));
    }

    // ---- 2. gemv (n==1) for the dtype-agnostic backends (VectorJAM declines n==1; see below) ----

    @Test void scalarGemvEveryDtype() {
        for (int t : ALL) parity(SCALAR, "ScalarJAM.gemv " + name(t), t, 512, 1, blockK(t), tol(SCALAR));
    }

    @Test void nativeGemvEveryDtype() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int t : ALL) parity(NATIVE, "NativeJAM.gemv " + name(t), t, 512, 1, blockK(t), tol(NATIVE));
    }

    // ---- 2b. EXHAUSTIVE seq-length × shape sweep ----------------------------------------------------
    // The fixed n∈{7,8,13,16} cases above never exercise SHORT prefills (n∈2..6), and a tile/seq-remainder
    // bug hides exactly there: a backend can pick a different micro-kernel per (n mod tile) and per row-tile
    // remainder, reading uninitialized repack/requant scratch on some residues only. Every backend × every
    // dtype must match the double-precision reference at EVERY n in 1..17 and for row counts both aligned
    // (104, 64) and not (100, 70) to the kernel row tile. Regression for the Q8_0 native cached-repack
    // seq-remainder family (the n∈[2,8) path on AVX-512-VNNI + AVX-VNNI CPUs).

    /** (m, k): m both 8/16-tile-aligned (104, 64) and not (100, 70); every k a multiple of 256 (valid for all dtypes). */
    private static final int[][] SWEEP_SHAPES = {{104, 1024}, {100, 512}, {70, 256}, {64, 768}};
    private static final int MAX_SEQ = 17;

    @Test void seqShapeSweepScalar() {
        for (int[] mk : SWEEP_SHAPES) for (int n = 1; n <= MAX_SEQ; n++) for (int t : ALL)
            parity(SCALAR, "ScalarJAM " + name(t) + " m=" + mk[0] + " n=" + n, t, mk[0], n, mk[1], tol(SCALAR));
    }

    @Test void seqShapeSweepVector() {                       // VectorJAM declines n==1 (decode) -> n from 2
        for (int[] mk : SWEEP_SHAPES) for (int n = 2; n <= MAX_SEQ; n++) for (int t : TILEABLE)
            parity(VECTOR, "VectorJAM " + name(t) + " m=" + mk[0] + " n=" + n, t, mk[0], n, mk[1], tol(VECTOR));
    }

    @Test void seqShapeSweepNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int[] mk : SWEEP_SHAPES) for (int n = 1; n <= MAX_SEQ; n++) for (int t : ALL)
            parity(NATIVE, "NativeJAM " + name(t) + " m=" + mk[0] + " n=" + n, t, mk[0], n, mk[1], tol(NATIVE));
    }

    /** Native self-consistency: the SAME inputs run twice must give bit-identical output. A difference is an
     *  uninitialized-scratch read (the cross-process non-determinism, surfaced in-process when the kernel
     *  re-reads fresh garbage). Complements the reference parity above, which catches a wrong-but-stable result. */
    @Test void nativeDeterministic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int[] mk : SWEEP_SHAPES) {
            int m = mk[0], k = mk[1];
            for (int n = 1; n <= MAX_SEQ; n++) for (int t : ALL) {
                Random rng = new Random(SEED ^ t ^ ((long) m << 20) ^ ((long) n << 8) ^ k ^ 0xDE7L);
                QuantWeights.Weight w = QuantWeights.encode(t, m, k, A, rng);
                float[] av = QuantWeights.gaussians(n * k, rng);
                MemorySegment a = A.allocate(av.length * 4L, 64);
                for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
                MemorySegment r1 = A.allocate((long) n * m * 4, 64), r2 = A.allocate((long) n * m * 4, 64);
                assertEquals(JAM.OK, run(NATIVE, w, a, r1, m, n, k), "native status");
                assertEquals(JAM.OK, run(NATIVE, w, a, r2, m, n, k), "native status");
                for (int i = 0; i < (long) n * m; i++)
                    assertEquals(r1.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 r2.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 0.0f, "NativeJAM non-deterministic: " + name(t) + " m=" + m + " n=" + n + " idx=" + i);
            }
        }
    }

    // ---- 2c. Non-zero weight OFFSET sweep (the MoE expert pattern) ---------------------------------
    // Every gemm above starts the weight at byte 0. A MoE expert gemm instead reads a slice of a packed
    // expert tensor at a non-zero (block-aligned) row offset — a code path the offset-0 cases never touch
    // (base-address arithmetic, repack-cache keying). Sweep a couple of offsets × the seq range so an
    // offset-only or offset×seq remainder bug can't hide.

    /** Like {@link #parity} but the weight is read starting at row {@code r0} of a taller encoded tensor. */
    private static void parityOffset(JAM backend, String name, int tag, int r0, int m, int n, int k, double relTol) {
        Random rng = new Random(SEED ^ tag ^ name.hashCode() ^ ((long) r0 << 32) ^ ((long) m << 20) ^ ((long) n << 8) ^ k);
        QuantWeights.Weight w = QuantWeights.encode(tag, r0 + m, k, A, rng);   // tall weight; gemm reads rows [r0, r0+m)
        long rowBytes = w.seg().byteSize() / (r0 + m);                          // contiguous -> exact per-row stride
        float[] av = QuantWeights.gaussians(n * k, rng);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment r = A.allocate((long) n * m * 4, 64);

        assertEquals(JAM.OK, backend.mm(w.seg(), (long) r0 * rowBytes, tag, k, a, 0, JAM.F32, k, r, 0, JAM.F32, m, m, n, k),
                name + " status");
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i += 7) {
            double[] ref = QuantWeights.refDot(w, av, r0 + i, j, k);            // reference row is r0+i in the tall weight
            float got = r.get(ValueLayout.JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
            assertEquals(ref[0], got, relTol * ref[1] + 1e-3, name + "[token " + j + ", row " + i + "]");
        }
    }

    private static final int[] ROW_OFFSETS = {8, 24};   // block-aligned (k % 256 == 0 makes any row offset aligned)

    @Test void seqShapeOffsetSweepNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int r0 : ROW_OFFSETS) for (int[] mk : SWEEP_SHAPES) for (int n = 1; n <= MAX_SEQ; n++) for (int t : ALL)
            parityOffset(NATIVE, "NativeJAM " + name(t) + " r0=" + r0 + " m=" + mk[0] + " n=" + n, t, r0, mk[0], n, mk[1], tol(NATIVE));
    }

    @Test void seqShapeOffsetSweepVector() {
        for (int r0 : ROW_OFFSETS) for (int[] mk : SWEEP_SHAPES) for (int n = 2; n <= MAX_SEQ; n++) for (int t : TILEABLE)
            parityOffset(VECTOR, "VectorJAM " + name(t) + " r0=" + r0 + " m=" + mk[0] + " n=" + n, t, r0, mk[0], n, mk[1], tol(VECTOR));
    }

    // ---- 2d. Large-m native DETERMINISM (the multi-threaded race) ----------------------------------
    // The model's real gemms have thousands of output rows, so jam fans them across many worker threads —
    // far more parallel work units than the m≤104 shapes above ever create. A race on shared worker state
    // surfaces only at that scale, and is non-deterministic (thread-count dependent: JAM_NUM_THREADS=1 is
    // clean). Two runs on IDENTICAL inputs must be bit-identical; a difference is the race. (Regression for
    // the jam-native threaded-gemm non-determinism observed in A4B Q8_0 prefill.)

    /** (m, k): model-scale row counts (many bands) × a few block-aligned k. */
    private static final int[][] BIG_SHAPES = {{2048, 2560}, {4096, 2048}, {2560, 1024}};

    @Test void nativeDeterministicLargeM() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        for (int[] mk : BIG_SHAPES) {
            int m = mk[0], k = mk[1];
            for (int n : new int[]{2, 4, 6, 7, 13}) for (int t : ALL) {
                Random rng = new Random(SEED ^ t ^ ((long) m << 20) ^ ((long) n << 8) ^ k ^ 0xB16L);
                QuantWeights.Weight w = QuantWeights.encode(t, m, k, A, rng);
                float[] av = QuantWeights.gaussians(n * k, rng);
                MemorySegment a = A.allocate(av.length * 4L, 64);
                for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
                MemorySegment r1 = A.allocate((long) n * m * 4, 64), r2 = A.allocate((long) n * m * 4, 64);
                assertEquals(JAM.OK, run(NATIVE, w, a, r1, m, n, k), "native status");
                assertEquals(JAM.OK, run(NATIVE, w, a, r2, m, n, k), "native status");
                for (long i = 0; i < (long) n * m; i++)
                    assertEquals(r1.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 r2.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 0.0f, "NativeJAM non-deterministic (race): " + name(t) + " m=" + m + " n=" + n + " idx=" + i);
            }
        }
    }

    /** Large-m AND non-zero weight offset, run twice — the MoE expert gemm is both (big + sliced). */
    @Test void nativeDeterministicLargeMOffset() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        int r0 = 24;
        for (int[] mk : BIG_SHAPES) {
            int m = mk[0], k = mk[1];
            for (int n : new int[]{2, 4, 6, 7, 13}) for (int t : ALL) {
                Random rng = new Random(SEED ^ t ^ ((long) m << 20) ^ ((long) n << 8) ^ k ^ 0xB16F5L);
                QuantWeights.Weight w = QuantWeights.encode(t, r0 + m, k, A, rng);
                long rowBytes = w.seg().byteSize() / (r0 + m);
                float[] av = QuantWeights.gaussians(n * k, rng);
                MemorySegment a = A.allocate(av.length * 4L, 64);
                for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
                MemorySegment r1 = A.allocate((long) n * m * 4, 64), r2 = A.allocate((long) n * m * 4, 64);
                long off = (long) r0 * rowBytes;
                assertEquals(JAM.OK, NATIVE.mm(w.seg(), off, t, k, a, 0, JAM.F32, k, r1, 0, JAM.F32, m, m, n, k), "status");
                assertEquals(JAM.OK, NATIVE.mm(w.seg(), off, t, k, a, 0, JAM.F32, k, r2, 0, JAM.F32, m, m, n, k), "status");
                for (long i = 0; i < (long) n * m; i++)
                    assertEquals(r1.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 r2.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 0.0f, "NativeJAM non-deterministic (race): " + name(t) + " r0=" + r0 + " m=" + m + " n=" + n + " idx=" + i);
            }
        }
    }

    // ---- 2e. The EXACT A4B prefill gemm shapes (from -Djinfer.mmTrace), run many times -------------
    // dim=2816. These are the shapes the live model issues at prefill (n=6) where native output is
    // non-deterministic run-to-run. Reproduces the bug from the real shapes (k=2816 — never swept above).
    private static final int[][] A4B_PREFILL_SHAPES = {
        {JAM.Q8_0, 2112, 2816},   // shared MLP gate/up
        {JAM.Q8_0, 2816, 2112},   // shared MLP down
        {JAM.F32,   128, 2816},   // MoE router (top-k sensitive)
        {JAM.Q8_0, 1408, 2816},   // expert gate-up
        {JAM.Q8_0, 2816,  704},   // expert down
        {JAM.Q8_0, 2048, 2816},   // attention Q (deterministic control)
        {JAM.Q8_0, 4096, 2816}, {JAM.Q8_0, 2816, 4096},
    };

    @Test void a4bPrefillShapesDeterministic() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        final int REPS = 8;
        for (int n : new int[]{2, 4, 6}) for (int[] sh : A4B_PREFILL_SHAPES) {
            int tag = sh[0], m = sh[1], k = sh[2];
            Random rng = new Random(SEED ^ tag ^ ((long) m << 20) ^ ((long) n << 8) ^ k ^ 0xA4B0L);
            QuantWeights.Weight w = QuantWeights.encode(tag, m, k, A, rng);
            float[] av = QuantWeights.gaussians(n * k, rng);
            MemorySegment a = A.allocate(av.length * 4L, 64);
            for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
            MemorySegment ref = A.allocate((long) n * m * 4, 64);
            assertEquals(JAM.OK, run(NATIVE, w, a, ref, m, n, k), "status");
            for (int rep = 1; rep < REPS; rep++) {
                MemorySegment r = A.allocate((long) n * m * 4, 64);
                assertEquals(JAM.OK, run(NATIVE, w, a, r, m, n, k), "status");
                for (long i = 0; i < (long) n * m; i++)
                    assertEquals(ref.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 r.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                                 0.0f, "NativeJAM non-deterministic: " + name(tag) + " " + m + "x" + k + " n=" + n + " rep=" + rep + " idx=" + i);
            }
        }
    }

    // ---- 2f. The MoE EXPERT gemm: read a packed [nExpert, gateUpDim, dim] tensor at a LARGE per-expert
    // offset (e·gateUpDim rows), the A4B layer-0 case the model trace localizes the non-determinism to
    // (shared-MLP + router at offset 0 are bit-stable; only these offset gemms diverge). Big offsets
    // (~0.5 GB byte) — never hit by the small-offset sweep above. Native vs the reference at each slice. ----
    @Test void moeExpertOffsetParityNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        int gateUpDim = 1408, dim = 2816;
        for (int e : new int[]{0, 1, 31, 64, 100, 127}) {
            int r0 = e * gateUpDim;                          // expert e's row offset into the packed tensor
            for (int n : new int[]{1, 2, 6})
                parityOffset(NATIVE, "NativeJAM expert e=" + e + " n=" + n, JAM.Q8_0, r0, gateUpDim, n, dim, tol(NATIVE));
        }
    }

    /** Cross-call determinism at the expert offset: the SAME large-offset gemm, run interleaved with other
     *  shapes (as the model does), must stay bit-identical. Reproduces the cross-call state dependence. */
    @Test void moeExpertOffsetDeterministicNative() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        int gateUpDim = 1408, dim = 2816, n = 6;
        for (int e : new int[]{1, 64, 127}) {
            int r0 = e * gateUpDim;
            Random rng = new Random(SEED ^ JAM.Q8_0 ^ ((long) r0 << 8) ^ dim ^ 0xE0FF5L);
            QuantWeights.Weight w = QuantWeights.encode(JAM.Q8_0, r0 + gateUpDim, dim, A, rng);
            long rowBytes = w.seg().byteSize() / (r0 + gateUpDim), off = (long) r0 * rowBytes;
            float[] av = QuantWeights.gaussians(n * dim, rng);
            MemorySegment a = A.allocate(av.length * 4L, 64);
            for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
            MemorySegment r1 = A.allocate((long) n * gateUpDim * 4, 64), r2 = A.allocate((long) n * gateUpDim * 4, 64);
            assertEquals(JAM.OK, NATIVE.mm(w.seg(), off, JAM.Q8_0, dim, a, 0, JAM.F32, dim, r1, 0, JAM.F32, gateUpDim, gateUpDim, n, dim), "status");
            // a differently-shaped gemm between the two (perturbs jam's reused scratch), as in the model
            QuantWeights.Weight w2 = QuantWeights.encode(JAM.Q8_0, 2816, 2112, A, rng);
            MemorySegment a2 = A.allocate(6L * 2112 * 4, 64), o2 = A.allocate(6L * 2816 * 4, 64);
            NATIVE.mm(w2.seg(), 0, JAM.Q8_0, 2112, a2, 0, JAM.F32, 2112, o2, 0, JAM.F32, 2816, 2816, 6, 2112);
            assertEquals(JAM.OK, NATIVE.mm(w.seg(), off, JAM.Q8_0, dim, a, 0, JAM.F32, dim, r2, 0, JAM.F32, gateUpDim, gateUpDim, n, dim), "status");
            for (long i = 0; i < (long) n * gateUpDim; i++)
                assertEquals(r1.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                             r2.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L),
                             0.0f, "expert-offset non-deterministic across calls: e=" + e + " idx=" + i);
        }
    }

    // ---- 2g. The rp path under the model's interleaved variable-n pattern -------------------------
    // The MoE prefill issues a fixed-n shared-MLP gemm, then many expert gemms with VARYING small n (the
    // tokens routed to each), all reusing one jam ctx's requant/repack scratch. The shared-MLP (n=6) is
    // bit-stable but the variable-n experts jitter in the live model. This drives that exact pattern —
    // varying n against a persistent ctx — and checks every result against the reference (guards the rp
    // path's correctness for the variable-n case; the bit-level non-determinism is caught model-side by
    // jinfer's PrefillDeterminism, which isolated jam calls can't reproduce).
    @Test void rpPathInterleavedVariableN() {
        org.junit.jupiter.api.Assumptions.assumeTrue(NATIVE != null, "libjam not loaded");
        int gateUpDim = 1408, dim = 2816, expertFF = 704;
        // (m, k) for the A4B FFN gemms the rp path serves at prefill
        int[][] ffn = {{2112, dim}, {dim, 2112}, {gateUpDim, dim}, {dim, expertFF}};
        for (int[] mk : ffn) {
            int m = mk[0], k = mk[1];
            // shared-MLP-style n=6 first, then the expert spread of small n, back-to-back on one ctx
            for (int n : new int[]{6, 1, 2, 1, 3, 1, 2, 4, 1, 5, 6, 1})
                parity(NATIVE, "NativeJAM rp " + m + "x" + k + " n=" + n, JAM.Q8_0, m, n, k, tol(NATIVE));
        }
    }

    // ---- 3. VectorJAM's decline contract: it must return EUNSUPPORTED (not a wrong answer) ----

    @Test void vectorDeclineContract() {
        Random rng = new Random(SEED ^ 0xdec1e5L);
        int m = 64, k = 256;
        QuantWeights.Weight q = QuantWeights.encode(JAM.Q8_0, m * 4, k, A, rng);  // enough rows for the strided probe
        MemorySegment x1 = QuantWeights.f32Row(k, A, rng), o1 = A.allocate((long) m * 4, 64);
        MemorySegment x4 = QuantWeights.f32Row(4 * k, A, rng), o4 = A.allocate((long) 4 * m * 4, 64);

        assertNotEquals(JAM.OK, VECTOR.mm(q.seg(), 0, JAM.Q8_0, k, x1, 0, JAM.F32, k, o1, 0, JAM.F32, m, m, 1, k),
                "VectorJAM must decline n==1 (decode)");
        QuantWeights.Weight wf = QuantWeights.encode(JAM.F32, m, k, A, rng);
        assertNotEquals(JAM.OK, VECTOR.mm(wf.seg(), 0, JAM.F32, k, x4, 0, JAM.F32, k, o4, 0, JAM.F32, m, m, 4, k),
                "VectorJAM must decline an F32 weight (no tile)");
        assertNotEquals(JAM.OK, VECTOR.mm(q.seg(), 0, JAM.Q8_0, k + 32, x4, 0, JAM.F32, k, o4, 0, JAM.F32, m, m, 4, k),
                "VectorJAM must decline a strided weight (ldw != k)");
        assertNotEquals(JAM.OK, VECTOR.mm(q.seg(), 0, JAM.Q8_0, k, x4, 0, JAM.F16, k, o4, 0, JAM.F32, m, m, 4, k),
                "VectorJAM must decline a non-F32 activation");
    }

    // ---- 4. cross-backend agreement: Vector ≈ Scalar (≈ Native) on identical inputs ----

    @Test void crossBackendAgreement() {
        for (int[] sh : new int[][]{{96, 12, 512}, {104, 16, 1024}, {64, 8, 256}, {128, 5, 768}}) {
            int m = sh[0], n = sh[1], k = sh[2];
            Random rng = new Random(SEED ^ 0xc0ffeeL ^ ((long) m << 16) ^ k);
            QuantWeights.Weight w = QuantWeights.encode(JAM.Q8_0, m, k, A, rng);
            float[] av = QuantWeights.gaussians(n * k, rng);
            MemorySegment a = A.allocate(av.length * 4L, 64);
            for (int i = 0; i < av.length; i++) a.set(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);

            MemorySegment rv = A.allocate((long) n * m * 4, 64), rs = A.allocate((long) n * m * 4, 64);
            assertEquals(JAM.OK, run(VECTOR, w, a, rv, m, n, k), "vector status");
            assertEquals(JAM.OK, run(SCALAR, w, a, rs, m, n, k), "scalar status");
            MemorySegment rn = NATIVE == null ? null : A.allocate((long) n * m * 4, 64);
            if (rn != null) assertEquals(JAM.OK, run(NATIVE, w, a, rn, m, n, k), "native status");

            for (int j = 0; j < n; j++) for (int row = 0; row < m; row++) {
                long i = (long) j * m + row;
                double tol = 1e-2 * QuantWeights.refDot(w, av, row, j, k)[1] + 1e-2;   // int8-quant error ∝ Σ|w·a|
                float sv = rs.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4);
                float vv = rv.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4);
                assertEquals(sv, vv, tol, "Vector vs Scalar [" + m + "x" + n + "x" + k + "] token=" + j + " row=" + row);
                if (rn != null) {
                    float nv = rn.get(ValueLayout.JAVA_FLOAT_UNALIGNED, i * 4);
                    assertEquals(sv, nv, tol, "Native vs Scalar [" + m + "x" + n + "x" + k + "] token=" + j + " row=" + row);
                }
            }
        }
    }

    // ---- 5. tile selection is a valid code and consistent with the vector width ----

    @Test void tileSelectionSane() {
        assertTrue(Q8Kernel.tileCode() >= 0 && Q8Kernel.tileCode() <= 12, "TILE_CODE in [0,12], was " + Q8Kernel.tileCode());
        assertTrue(!VectorSupport.IS_512 || Q8Kernel.tileCode() <= 5, "IS_512 implies a 512-bit tile code (<=5)");
    }

    // ---- helpers ----

    /** k a whole number of blocks for the dtype (256 for k-quants, also a multiple of 64/32 for FP4/Q*_0). */
    private static int blockK(int tag) {
        return switch (tag) {
            case JAM.Q4_K, JAM.Q5_K, JAM.Q6_K, JAM.NVFP4, JAM.MXFP4 -> 1024;   // multiple of 256/64/32
            default -> 1024;                                                    // F*/Q*_0: any multiple of 32
        };
    }

    private static String name(int tag) {
        return switch (tag) {
            case JAM.F32 -> "F32"; case JAM.F16 -> "F16"; case JAM.BF16 -> "BF16";
            case JAM.Q8_0 -> "Q8_0"; case JAM.Q4_0 -> "Q4_0"; case JAM.Q4_K -> "Q4_K";
            case JAM.Q5_K -> "Q5_K"; case JAM.Q6_K -> "Q6_K"; case JAM.MXFP4 -> "MXFP4"; case JAM.NVFP4 -> "NVFP4";
            default -> "tag" + tag;
        };
    }
}
