package com.qxotic.jam;

import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Standalone correctness for jam-vector's relocated SIMD kernels — so the jam reactor verifies its own
 * register-tiled gemm instead of relying on jinfer downstream. For each tileable dtype a synthetic
 * quantized weight (exact-representable, reusing ScalarJamTest's encoders) and a random F32 activation are
 * fed through BOTH the vector kernel and {@link ScalarJAM} (the complete reference floor); the two must
 * agree to within the int8-activation quantization the tiles use. Kernels are invoked exactly as jinfer
 * does — weight as a raw segment, activation + output through the GLOBAL segment at their absolute
 * addresses — which is uniform whether a kernel stores via {@code o.set} or absolute {@code putFloat}.
 *
 * <p>The kernels need a 128/256/512-bit FloatVector; the suite is skipped (not failed) where that's absent.
 */
class VectorKernelTest {

    private static final ScalarJAM SCALAR = new ScalarJAM();
    private static final Arena A = Arena.ofAuto();
    private static final Random RNG = new Random(11);

    /** A vector kernel's public gemm entry (register-tiled dtypes; the band kernels add a trailing Scratch). */
    @FunctionalInterface
    interface Gemm {
        void run(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                 int aStride, int oStride, int n, int m, int k, long wOff);
    }

    /** A band kernel's gemm entry (k-quants, FP4): same as {@link Gemm} plus the context-owned dequant pool. */
    @FunctionalInterface
    interface BandGemm {
        void run(MemorySegment w, MemorySegment a, long aBase, MemorySegment o, long oBase,
                 int aStride, int oStride, int n, int m, int k, long wOff, Scratch scratch);
    }

    /** Adapt a band kernel to {@link Gemm} by binding a fresh per-test {@link Scratch} (as a real context would). */
    private static Gemm withScratch(BandGemm g) {
        Scratch s = new Scratch();
        return (w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff)
                -> g.run(w, a, aBase, o, oBase, aStride, oStride, n, m, k, wOff, s);
    }

    @Test void q8_0()  { eachShape("Q8_0",  VectorKernelTest::q8_0,  Q8Kernel::gemm); }
    @Test void q4_0()  { eachShape("Q4_0",  VectorKernelTest::q4_0,  Q4Kernel::gemm); }
    @Test void q4_k()  { eachShape("Q4_K",  VectorKernelTest::q4_k,  withScratch(Q4KKernel::gemm)); }
    @Test void q5_k()  { eachShape("Q5_K",  VectorKernelTest::q5_k,  withScratch(Q5KKernel::gemm)); }
    @Test void q6_k()  { eachShape("Q6_K",  VectorKernelTest::q6_k,  withScratch(Q6KKernel::gemm)); }
    @Test void mxfp4() { eachShape("MXFP4", VectorKernelTest::mxfp4, withScratch(Mxfp4Kernel::gemm)); }
    @Test void nvfp4() { eachShape("NVFP4", VectorKernelTest::nvfp4, withScratch(Nvfp4Kernel::gemm)); }

    /** Check one dtype's kernel against ScalarJAM at n in {8,13,16} (full tile + both remainders). */
    private static void eachShape(String name, java.util.function.BiFunction<Integer, Integer, Weight> encode, Gemm kernel) {
        Assumptions.assumeTrue(VectorSupport.F_SPECIES.vectorBitSize() >= 128, "vector kernels require a >=128-bit FloatVector");
        int m = 104, k = 256;                       // k = one k-quant super-block (also a multiple of 64/32)
        for (int n : new int[]{8, 13, 16}) check(name + " n=" + n, encode.apply(m, k), kernel, m, n, k);
    }

    /** Run the vector kernel and ScalarJAM on the same inputs; assert agreement within int8-activation error. */
    private static void check(String name, Weight w, Gemm kernel, int m, int n, int k) {
        float[] av = gaussians(n * k);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment ov = A.allocate((long) n * m * 4, 64);   // vector output
        MemorySegment os = A.allocate((long) n * m * 4, 64);   // scalar reference output

        // vector: invoked as jinfer does — weight raw, activation/output via GLOBAL at absolute addresses.
        kernel.run(w.seg(), VectorSupport.GLOBAL, a.address(), VectorSupport.GLOBAL, ov.address(),
                   k, m, n, m, k, 0L);
        // scalar reference: segment-relative.
        int st = SCALAR.mm(w.seg(), 0, w.tag(), k, a, 0, JAM.F32, k, os, 0, JAM.F32, m, m, n, k);
        assertEquals(JAM.OK, st, name + " scalar status");

        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
            double sumAbs = 0;                                 // int8 error scales with sum|w*a|, not |value|
            for (int l = 0; l < k; l++) sumAbs += Math.abs((double) w.vals()[i * k + l] * av[j * k + l]);
            float vv = ov.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
            float sv = os.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
            assertEquals(sv, vv, sumAbs * 1e-2 + 1e-3, name + "[token " + j + ", row " + i + "]");
        }
    }

    // ---- weight: dtype tag, encoded segment, exact decoded values (encoders copied from ScalarJamTest) ----

    private record Weight(int tag, MemorySegment seg, float[] vals) {}

    /** E2M1 FP4 code -> signed magnitude (MXFP4/NVFP4 nibble table). */
    private static final int[] FP4_KV = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    private static Weight q8_0(int m, int k) {              // scale 1.0, int8 qs -> exact integer values
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 34, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 34;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            for (int e = 0; e < 32; e++) {
                byte q = (byte) (RNG.nextInt(255) - 127);
                s.set(JAVA_BYTE, blk + 2 + e, q);
                v[r * k + b * 32 + e] = q;
            }
        }
        return new Weight(JAM.Q8_0, s, v);
    }

    private static Weight q4_0(int m, int k) {              // scale 1.0, nibble -> value (nibble-8) in [-8,7]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 18, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 18;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            for (int e = 0; e < 16; e++) {
                int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                s.set(JAVA_BYTE, blk + 2 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 32 + e] = lo - 8;
                v[r * k + b * 32 + 16 + e] = hi - 8;
            }
        }
        return new Weight(JAM.Q4_0, s, v);
    }

    private static Weight q4_k(int m, int k) {       // 144B/256-elem; d=dmin=1 -> value = sc*nibble - mn
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 144, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 144;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f));
            int[] sc = small(8), mn = small(8);
            packScalesK4(s, blk + 4, sc, mn);
            for (int g = 0; g < 4; g++) for (int e = 0; e < 32; e++) {
                int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                s.set(JAVA_BYTE, blk + 16 + g * 32 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 256 + g * 64 + e]      = sc[g * 2]     * lo - mn[g * 2];
                v[r * k + b * 256 + g * 64 + 32 + e] = sc[g * 2 + 1] * hi - mn[g * 2 + 1];
            }
        }
        return new Weight(JAM.Q4_K, s, v);
    }

    private static Weight q5_k(int m, int k) {       // 176B/256-elem; 5-bit q = nibble | (qh bit<<4)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 176, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 176;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f));
            int[] sc = small(8), mn = small(8);
            packScalesK4(s, blk + 4, sc, mn);
            int[] qh = new int[32];
            for (int g = 0; g < 4; g++) for (int e = 0; e < 32; e++) {
                int lo = RNG.nextInt(16), hi = RNG.nextInt(16), bLo = RNG.nextInt(2), bHi = RNG.nextInt(2);
                qh[e] |= (bLo << (2 * g)) | (bHi << (2 * g + 1));
                s.set(JAVA_BYTE, blk + 48 + g * 32 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 256 + g * 64 + e]      = sc[g * 2]     * (lo | (bLo << 4)) - mn[g * 2];
                v[r * k + b * 256 + g * 64 + 32 + e] = sc[g * 2 + 1] * (hi | (bHi << 4)) - mn[g * 2 + 1];
            }
            for (int e = 0; e < 32; e++) s.set(JAVA_BYTE, blk + 16 + e, (byte) qh[e]);
        }
        return new Weight(JAM.Q5_K, s, v);
    }

    private static Weight q6_k(int m, int k) {       // 210B/256-elem; d=1 -> value = sc*(qv-32)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 210, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 210;
            int[] ql = new int[128], qh = new int[64], sc = small(16);
            for (int h = 0; h < 2; h++) for (int j = 0; j < 4; j++) for (int ll = 0; ll < 32; ll++) {
                int qv = RNG.nextInt(64);
                int qlIdx = h * 64 + ((j == 1 || j == 3) ? 32 + ll : ll);
                if (j < 2) ql[qlIdx] |= qv & 0xF; else ql[qlIdx] |= (qv & 0xF) << 4;
                qh[h * 32 + ll] |= ((qv >> 4) & 3) << (2 * j);
                v[r * k + b * 256 + h * 128 + j * 32 + ll] = sc[h * 8 + j * 2 + ll / 16] * (qv - 32);
            }
            for (int t = 0; t < 128; t++) s.set(JAVA_BYTE, blk + t, (byte) ql[t]);
            for (int t = 0; t < 64; t++)  s.set(JAVA_BYTE, blk + 128 + t, (byte) qh[t]);
            for (int t = 0; t < 16; t++)  s.set(JAVA_BYTE, blk + 192 + t, (byte) sc[t]);
            s.set(JAVA_SHORT_UNALIGNED, blk + 208, floatToF16(1f));
        }
        return new Weight(JAM.Q6_K, s, v);
    }

    private static Weight mxfp4(int m, int k) {      // 17B/32-elem; e=128 -> dhalf=1 -> value = kv[nibble]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 17, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 17;
            s.set(JAVA_BYTE, blk, (byte) 128);
            for (int t = 0; t < 16; t++) {
                int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                s.set(JAVA_BYTE, blk + 1 + t, (byte) (lo | (hi << 4)));
                v[r * k + b * 32 + t]      = FP4_KV[lo];
                v[r * k + b * 32 + 16 + t] = FP4_KV[hi];
            }
        }
        return new Weight(JAM.MXFP4, s, v);
    }

    private static Weight nvfp4(int m, int k) {      // 36B/64-elem; d=0x38 -> ue4m3=1 -> value = kv[nibble]
        int nb = k / 64;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 36, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 36;
            for (int sub = 0; sub < 4; sub++) {
                s.set(JAVA_BYTE, blk + sub, (byte) 0x38);
                for (int j = 0; j < 8; j++) {
                    int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                    s.set(JAVA_BYTE, blk + 4 + sub * 8 + j, (byte) (lo | (hi << 4)));
                    v[r * k + b * 64 + sub * 16 + j]     = FP4_KV[lo];
                    v[r * k + b * 64 + sub * 16 + 8 + j] = FP4_KV[hi];
                }
            }
        }
        return new Weight(JAM.NVFP4, s, v);
    }

    // ---- helpers (copied from ScalarJamTest) ----

    /** {@code n} small non-negative ints (0..7) — keeps scale*quant products exact. */
    private static int[] small(int n) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) x[i] = RNG.nextInt(8);
        return x;
    }

    /** Pack sc[8]/mn[8] into the 12-byte k-quant scales block (inverse of get_scale_min_k4). */
    private static void packScalesK4(MemorySegment s, long base, int[] sc, int[] mn) {
        for (int j = 0; j < 4; j++) {
            s.set(JAVA_BYTE, base + j,     (byte) ((sc[j] & 63)      | ((sc[j + 4] >> 4) << 6)));
            s.set(JAVA_BYTE, base + j + 4, (byte) ((mn[j] & 63)      | ((mn[j + 4] >> 4) << 6)));
            s.set(JAVA_BYTE, base + j + 8, (byte) ((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4)));
        }
    }

    private static float[] gaussians(int n) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) RNG.nextGaussian();
        return v;
    }

    /** float -> IEEE half (round toward zero on the mantissa; enough for test values). */
    private static short floatToF16(float f) {
        int x = Float.floatToRawIntBits(f);
        int sign = (x >>> 16) & 0x8000;
        int e = ((x >>> 23) & 0xFF) - 127 + 15;
        int man = x & 0x7FFFFF;
        if (e <= 0) return (short) sign;
        if (e >= 31) return (short) (sign | 0x7C00);
        return (short) (sign | (e << 10) | (man >> 13));
    }
}
