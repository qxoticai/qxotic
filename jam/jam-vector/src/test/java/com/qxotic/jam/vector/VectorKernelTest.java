package com.qxotic.jam.vector;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jam.JAM;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Standalone correctness for jam-vector's relocated SIMD kernels - so the jam reactor verifies its
 * own register-tiled gemm instead of relying on jinfer downstream. For each tileable dtype a
 * synthetic quantized weight (exact-representable, via the shared {@link QuantWeights} fixture) and
 * a random F32 activation are fed through both the vector kernel and the scalar reference; the two
 * must agree to within the int8-activation quantization the tiles use. Kernels are invoked exactly
 * as jinfer does - weight as a raw segment, activation + output through the GLOBAL segment at their
 * absolute addresses - which is uniform whether a kernel stores via {@code o.set} or absolute
 * {@code putFloat}.
 *
 * <p>The kernels need a 128/256/512-bit FloatVector; the suite is skipped (not failed) where that's
 * absent.
 */
class VectorKernelTest {

    private static final JAM SCALAR =
            JAM.providers().stream()
                    .filter(provider -> provider.id().equals("scalar"))
                    .findFirst()
                    .orElseThrow()
                    .create();
    private static final Arena A = Arena.ofAuto();
    private static final Random RNG = new Random(11);

    /**
     * A vector kernel's public gemm entry (register-tiled dtypes; the band kernels add a trailing
     * Scratch).
     */
    @FunctionalInterface
    interface Gemm {
        void run(
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
                long wOff);
    }

    /**
     * A band kernel's gemm entry (k-quants, FP4): same as {@link Gemm} plus the context-owned
     * dequant pool.
     */
    @FunctionalInterface
    interface BandGemm {
        void run(
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
                Scratch scratch);
    }

    /**
     * Adapt a band kernel to {@link Gemm} by binding a fresh per-test {@link Scratch} (as a real
     * context would).
     */
    private static Gemm withScratch(BandGemm g) {
        Scratch s = new Scratch();
        return (w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff) ->
                g.run(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, s);
    }

    @Test
    void q8_0() {
        eachShape("Q8_0", JAM.Q8_0, Q8Kernel::gemm);
    }

    @Test
    void q4_0() {
        eachShape("Q4_0", JAM.Q4_0, withScratch(Q4Kernel::gemm));
    }

    @Test
    void q4_k() {
        eachShape("Q4_K", JAM.Q4_K, withScratch(Q4KKernel::gemm));
    }

    @Test
    void q5_k() {
        eachShape("Q5_K", JAM.Q5_K, withScratch(Q5KKernel::gemm));
    }

    @Test
    void q6_k() {
        eachShape("Q6_K", JAM.Q6_K, withScratch(Q6KKernel::gemm));
    }

    @Test
    void mxfp4() {
        eachShape("MXFP4", JAM.MXFP4, withScratch(Mxfp4Kernel::gemm));
    }

    @Test
    void nvfp4() {
        eachShape("NVFP4", JAM.NVFP4, withScratch(Nvfp4Kernel::gemm));
    }

    @Test
    void q1_0() {
        eachShape("Q1_0", JAM.Q1_0, withScratch(Q1Kernel::gemm));
    }

    /**
     * Check one dtype's kernel against ScalarJAM at n in {8,13,16} (full tile + both remainders).
     */
    /**
     * The band gemm's blocking: k cut into kc-blocks accumulating into the output, panels of
     * several bands, a trailing partial band and a trailing partial tile - on a k that is neither
     * one block nor a multiple of the panel.
     */
    /**
     * The band gemm's blocking, for every dequant-to-scratch dtype: k cut into kc-blocks
     * accumulating into the output (each block's dequant starts at a non-zero element offset),
     * panels of several bands, a trailing partial band and a trailing partial tile - on a k that is
     * neither one block nor a multiple of the panel.
     */
    @Test
    void bandBlocking() {
        Assumptions.assumeTrue(VectorSupport.F_SPECIES.vectorBitSize() >= 128);
        Scratch s = new Scratch();
        int m = 61, k = 768; // 3 k-blocks of 256; m not a multiple of MR
        record Deq(String name, int tag, com.qxotic.jam.vector.BandGemm.RowDequant deq) {}
        Deq[] deqs = {
            new Deq("Q8_0", JAM.Q8_0, Q8Kernel::dequantizeRow),
            new Deq("Q4_0", JAM.Q4_0, Q4Kernel::dequantizeRow),
            new Deq("Q4_K", JAM.Q4_K, Q4KKernel::dequantizeRow),
            new Deq("Q5_K", JAM.Q5_K, Q5KKernel::dequantizeRow),
            new Deq("Q6_K", JAM.Q6_K, Q6KKernel::dequantizeRow),
        };
        for (Deq d : deqs)
            for (int n : new int[] {1, 7, 16, 50}) {
                Gemm blocked =
                        (w, a, aBase, o, oBase, aStride, oStride, nn, mm, kk, wOff) ->
                                com.qxotic.jam.vector.BandGemm.gemm(
                                        w, a, aBase, o, oBase, aStride, oStride, nn, mm, kk, wOff,
                                        s, d.deq(), 256);
                check(
                        d.name() + " blocked n=" + n,
                        QuantWeights.encode(d.tag(), m, k, A, RNG),
                        blocked,
                        m,
                        n,
                        k);
            }
    }

    private static void eachShape(String name, int tag, Gemm kernel) {
        Assumptions.assumeTrue(
                VectorSupport.F_SPECIES.vectorBitSize() >= 128,
                "vector kernels require a >=128-bit FloatVector");
        int m = 104, k = 256; // k = one k-quant super-block (also a multiple of 64/32)
        for (int n : new int[] {8, 13, 16})
            check(name + " n=" + n, QuantWeights.encode(tag, m, k, A, RNG), kernel, m, n, k);
    }

    /**
     * Run the vector kernel and ScalarJAM on the same inputs; assert agreement within
     * int8-activation error.
     */
    private static void check(
            String name, QuantWeights.Weight w, Gemm kernel, int m, int n, int k) {
        float[] av = QuantWeights.gaussians(n * k, RNG);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment ov = A.allocate((long) n * m * 4, 64); // vector output
        MemorySegment os = A.allocate((long) n * m * 4, 64); // scalar reference output

        // vector: invoked as jinfer does - weight raw, activation/output via GLOBAL at absolute
        // addresses.
        kernel.run(
                w.seg(),
                VectorSupport.GLOBAL,
                a.address(),
                VectorSupport.GLOBAL,
                ov.address(),
                k,
                m,
                n,
                m,
                k,
                0L);
        // scalar reference: segment-relative.
        int st = SCALAR.mm(w.seg(), 0, w.tag(), k, a, 0, JAM.F32, k, os, 0, JAM.F32, m, m, n, k);
        assertEquals(JAM.OK, st, name + " scalar status");

        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double sumAbs = 0; // int8 error scales with sum|w*a|, not |value|
                for (int l = 0; l < k; l++)
                    sumAbs += Math.abs((double) w.vals()[i * k + l] * av[j * k + l]);
                float vv = ov.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
                float sv = os.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
                assertEquals(
                        sv, vv, sumAbs * 1e-2 + 1e-3, name + "[token " + j + ", row " + i + "]");
            }
    }
}
