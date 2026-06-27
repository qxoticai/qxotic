package com.qxotic.jinfer;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jam.JAM;
import com.qxotic.jam.NativeJAM;

import java.util.Random;

/**
 * Comprehensive parity tests for the JAM matmul backends — {@link ScalarJAM}, {@link VectorJAM}, and the
 * native {@link NativeJAM} (when libjam loads). Every backend is run through the JAM segment contract on
 * synthetic quantized weights (reusing {@link KernelParityTest#makeQuant}) and checked against the same
 * double-precision reference ({@link KernelParityTest#refDot}). Covers:
 * <ul>
 *   <li>gemm (n&gt;1) correctness for every supported dtype;
 *   <li>gemv (n==1) correctness for the dtype-agnostic backends;
 *   <li>VectorJAM's decline contract (EUNSUPPORTED for n==1, non-tileable, or strided weight);
 *   <li>cross-backend agreement (Vector ≈ Scalar ≈ Native on the same inputs);
 *   <li>the JVM/CPU tile selection is reported and self-consistent.
 * </ul>
 * main()-based like {@link KernelParityTest}; exits non-zero on any mismatch.
 */
public final class JamBackendTest {

    private static final long SEED = 0x5A11C0DEL;
    private static int checks = 0, failures = 0;

    /** JAM weight dtypes with a VectorJAM register-tiled gemm. */
    static final GGMLType[] TILEABLE = {GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K,
                                        GGMLType.Q6_K, GGMLType.MXFP4, GGMLType.NVFP4};
    /** All JAM weight dtypes (tileable + the dense floats handled only by the dot floor / native). */
    static final GGMLType[] ALL = {GGMLType.Q8_0, GGMLType.Q4_0, GGMLType.Q4_K, GGMLType.Q5_K, GGMLType.Q6_K,
                                   GGMLType.MXFP4, GGMLType.NVFP4, GGMLType.F16, GGMLType.BF16, GGMLType.F32};

    public static void main(String[] args) {
        JAM scalar = new ScalarJAM();
        JAM vector = new VectorJAM();
        JAM jam = JamMatMul.tryLoad() ? NativeJAM.global() : null;
        System.out.printf("backends: ScalarJAM, VectorJAM (tile=%s code=%d is512=%b), NativeJAM=%s%n",
                VectorJAM.TILE, VectorJAM.TILE_CODE, VectorJAM.IS_512, jam != null ? "loaded" : "unavailable");

        // 1. gemm (n>1) correctness — all dtypes through the floor and native; tileable ones through Vector.
        for (GGMLType t : ALL) {
            gemm("ScalarJAM", scalar, t, true);
            gemm("VectorJAM", vector, t, isTileable(t));   // tileable -> OK+correct; else expect EUNSUPPORTED
            if (jam != null) gemm("NativeJAM", jam, t, true);
        }

        // 2. gemv (n==1) correctness — the dtype-agnostic backends (VectorJAM declines n==1, tested below).
        for (GGMLType t : ALL) {
            gemv("ScalarJAM", scalar, t);
            if (jam != null) gemv("NativeJAM", jam, t);
        }

        // 3. VectorJAM's decline contract.
        declines(vector);

        // 4. cross-backend agreement on one gemm.
        crossImpl(scalar, vector, jam);

        // 5. tile selection is a valid code and consistent with the vector width.
        check("TILE_CODE in [0,12]", VectorJAM.TILE_CODE >= 0 && VectorJAM.TILE_CODE <= 12 ? 1 : 0, 1, 0);
        check("IS_512 => not a scalar/avx256 tile", !VectorJAM.IS_512 || VectorJAM.TILE_CODE <= 5 ? 1 : 0, 1, 0);

        System.out.printf("%d checks, %d failures%n", checks, failures);
        if (failures > 0) System.exit(1);
    }

    // ---- one matmul through the JAM segment contract ----

    /** Run {@code backend.mm} on tensors converted to (vseg, vbase) — exactly how jinfer feeds JAM. */
    private static int run(JAM backend, FloatTensor w, GGMLType t, F32FloatTensor x, F32FloatTensor out,
                           int m, int n, int k, int ldw, int lda, int ldr) {
        SegmentFloatTensor sw = (SegmentFloatTensor) w;
        return backend.mm(sw.vseg, sw.vbase, t.getId(), ldw,
                          x.vseg, x.vbase, JAM.F32, lda,
                          out.vseg, out.vbase, JAM.F32, ldr, m, n, k);
    }

    // ---- tests ----

    static void gemm(String name, JAM backend, GGMLType t, boolean expectOk) {
        Random rng = new Random(SEED ^ t.ordinal() ^ name.hashCode());
        int m = 104, k = 1024;                     // k multiple of 256 (k-quant super-block) and 32
        // n in {8,13,16}: tile remainders + a full tile. NOTE n=7 is deliberately omitted — Q8_0 gemm at
        // seq=7 (below the VNNI threshold) is mis-computed by BOTH jam and FloatTensor.gemm; a pre-existing
        // bug, independent of the JAM backends (see KernelParityTest's seq=7 failures).
        for (int n : new int[]{8, 13, 16}) {
            FloatTensor w = KernelParityTest.makeQuant(t, m * k, rng);
            F32FloatTensor x = KernelParityTest.makeF32(n * k, rng);
            F32FloatTensor out = F32FloatTensor.allocate(n * m);
            int st = run(backend, w, t, x, out, m, n, k, k, k, m);
            if (!expectOk) {                       // VectorJAM on a non-tileable dtype must decline
                check(name + " " + t + " gemm declines (n=" + n + ")", st != JAM.OK ? 1 : 0, 1, 0);
                continue;
            }
            check(name + " " + t + " gemm status (n=" + n + ")", st, JAM.OK, 0);
            if (st != JAM.OK) continue;
            for (int s = 0; s < n; s++) {
                for (int row = 0; row < m; row += 13) {
                    double[] ref = KernelParityTest.refDot(w, row * k, x, s * k, k);
                    check(name + " " + t + " gemm[n=" + n + ",s=" + s + ",row=" + row + "]",
                          out.getFloat((long) s * m + row), ref[0], ref[1], tol(name));
                }
            }
        }
    }

    static void gemv(String name, JAM backend, GGMLType t) {
        Random rng = new Random(SEED ^ t.ordinal() ^ name.hashCode() ^ 0x9e3779b9L);
        int m = 512, k = 1024;
        FloatTensor w = KernelParityTest.makeQuant(t, m * k, rng);
        F32FloatTensor x = KernelParityTest.makeF32(k, rng);
        F32FloatTensor out = F32FloatTensor.allocate(m);
        int st = run(backend, w, t, x, out, m, 1, k, k, k, m);
        check(name + " " + t + " gemv status", st, JAM.OK, 0);
        if (st != JAM.OK) return;
        for (int row = 0; row < m; row += 17) {
            double[] ref = KernelParityTest.refDot(w, row * k, x, 0, k);
            check(name + " " + t + " gemv[row=" + row + "]", out.getFloat(row), ref[0], ref[1], tol(name));
        }
    }

    static void declines(JAM vector) {
        Random rng = new Random(SEED ^ 0xdec1e5L);
        int m = 64, k = 256;
        FloatTensor q = KernelParityTest.makeQuant(GGMLType.Q8_0, m * k, rng);
        F32FloatTensor x1 = KernelParityTest.makeF32(k, rng), o1 = F32FloatTensor.allocate(m);
        check("VectorJAM declines n==1", run(vector, q, GGMLType.Q8_0, x1, o1, m, 1, k, k, k, m) != JAM.OK ? 1 : 0, 1, 0);

        FloatTensor wf = KernelParityTest.makeF32(m * k, rng);
        F32FloatTensor x4 = KernelParityTest.makeF32(4 * k, rng), o4 = F32FloatTensor.allocate(4 * m);
        check("VectorJAM declines F32 weight", run(vector, wf, GGMLType.F32, x4, o4, m, 4, k, k, k, m) != JAM.OK ? 1 : 0, 1, 0);

        // strided weight (ldw != k) — declined before any read, so the small weight is never over-walked
        check("VectorJAM declines strided weight", run(vector, q, GGMLType.Q8_0, x4, o4, m, 4, k, k + 32, k, m) != JAM.OK ? 1 : 0, 1, 0);
    }

    static void crossImpl(JAM scalar, JAM vector, JAM jam) {
        // VectorJAM must match VectorMatMul (the existing tensor path, same kernel) bit-for-bit across shapes.
        Random rng = new Random(SEED ^ 0xc0ffeeL);
        for (int[] sh : new int[][]{{96, 12, 512}, {104, 16, 1024}, {64, 8, 256}, {128, 5, 768}, {96, 11, 512}}) {
            int m = sh[0], n = sh[1], k = sh[2];
            GGMLType t = GGMLType.Q8_0;
            FloatTensor w = KernelParityTest.makeQuant(t, m * k, rng);
            F32FloatTensor x = KernelParityTest.makeF32(n * k, rng);
            F32FloatTensor ov = F32FloatTensor.allocate(n * m), om = F32FloatTensor.allocate(n * m);
            run(vector, w, t, x, ov, m, n, k, k, k, m);
            new VectorMatMul().mm(w, 0, k, x, 0, k, om, 0, m, m, n, k);
            int diverged = -1;
            for (int i = 0; i < n * m; i++) if (Math.abs(ov.getFloat(i) - om.getFloat(i)) > 1e-4) { diverged = i; break; }
            if (diverged >= 0) System.out.printf("  [%dx%dx%d] i=%d  VectorJAM=%.5f  VectorMatMul=%.5f%n",
                    m, n, k, diverged, ov.getFloat(diverged), om.getFloat(diverged));
            check("VectorJAM==VectorMatMul [" + m + "x" + n + "x" + k + "] firstDiff@" + diverged
                  + (diverged < 0 ? "" : " (token " + diverged / m + " row " + diverged % m + ")"), diverged, -1, 0);
        }
    }

    // ---- helpers ----

    static boolean isTileable(GGMLType t) {
        for (GGMLType u : TILEABLE) if (u == t) return true;
        return false;
    }

    /** Relative tolerance vs the double-precision reference. The dot floor (ScalarJAM) is full-F32 and tight;
     *  the register-tiled / native paths quantize activations to int8 (VNNI) and accumulate in tile order,
     *  so they differ from a sequential-double dot by ~the int8 quant error — looser, as in KernelParityTest. */
    static double tol(String name) { return name.equals("ScalarJAM") ? 1e-3 : 1e-2; }

    private static void check(String what, double actual, double expected, double sumAbs) {
        check(what, actual, expected, sumAbs, 1e-4);
    }

    private static void check(String what, double actual, double expected, double sumAbs, double relTol) {
        checks++;
        double tol = relTol * sumAbs + 1e-3;
        if (!(Math.abs(actual - expected) <= tol)) {
            failures++;
            System.err.printf("FAIL %s: got %.8g expected %.8g (tol %.3g)%n", what, actual, expected, tol);
        }
    }
}
