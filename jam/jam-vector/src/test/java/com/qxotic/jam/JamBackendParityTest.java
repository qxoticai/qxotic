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
        assertTrue(!VectorJAM.IS_512 || Q8Kernel.tileCode() <= 5, "IS_512 implies a 512-bit tile code (<=5)");
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
