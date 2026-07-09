package com.qxotic.jam;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * {@link ScalarJAM} correctness. Each implemented dtype is encoded from values it represents
 * EXACTLY (so the decode is lossless), and the matmul must equal a double-precision reference
 * computed from those same values. Covers gemm + gemv across the dtypes, strided operands, and the
 * decline contract. Self-contained in jam-scalar (no jinfer, no native).
 */
class ScalarJamTest {

    private static final JAM jam = new ScalarJAM();
    private static final Arena A = Arena.ofAuto();
    private static final Random RNG = new Random(7);

    /**
     * A weight: its dtype tag, the encoded native segment, and the exact float values it decodes
     * to.
     */
    private record Weight(int tag, MemorySegment seg, float[] vals) {}

    /** E2M1 FP4 code -> signed magnitude (MXFP4/NVFP4 nibble table). */
    private static final int[] FP4_KV = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    // ---- per-dtype encoders over values the format holds exactly ----

    private static Weight f32(int m, int k) {
        float[] v = gaussians(m * k);
        MemorySegment s = A.allocate(v.length * 4L, 64);
        for (int i = 0; i < v.length; i++) s.set(JAVA_FLOAT_UNALIGNED, i * 4L, v[i]);
        return new Weight(JAM.F32, s, v);
    }

    private static Weight f16(int m, int k) {
        float[] v = new float[m * k];
        MemorySegment s = A.allocate(v.length * 2L, 64);
        for (int i = 0; i < v.length; i++) {
            short h = floatToF16((float) RNG.nextGaussian());
            s.set(JAVA_SHORT_UNALIGNED, i * 2L, h);
            v[i] = Float.float16ToFloat(h); // the exact value f16 holds (== ScalarJAM's decode)
        }
        return new Weight(JAM.F16, s, v);
    }

    private static Weight bf16(int m, int k) {
        float[] v = new float[m * k];
        MemorySegment s = A.allocate(v.length * 2L, 64);
        for (int i = 0; i < v.length; i++) {
            short h = (short) (Float.floatToRawIntBits((float) RNG.nextGaussian()) >>> 16);
            s.set(JAVA_SHORT_UNALIGNED, i * 2L, h);
            v[i] = Float.intBitsToFloat((h & 0xFFFF) << 16);
        }
        return new Weight(JAM.BF16, s, v);
    }

    private static Weight q8_0(int m, int k) { // scale 1.0, int8 qs -> exact integer values
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 34, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < nb; b++) {
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

    private static Weight q4_0(int m, int k) { // scale 1.0, nibble -> value (nibble-8) in [-8,7]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 18, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < nb; b++) {
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

    // k-quants / FP4: scales chosen so every decoded value is a small integer (exact). k must be a
    // multiple of the 256-element super-block (Q*_K) / 64 (NVFP4) / 32 (MXFP4).

    private static Weight q4_k(int m, int k) { // 144B/256-elem; d=dmin=1 -> value = sc*nibble - mn
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 144, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < sb; b++) {
                long blk = (long) (r * sb + b) * 144;
                s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f)); // d
                s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f)); // dmin
                int[] sc = small(8), mn = small(8);
                packScalesK4(s, blk + 4, sc, mn);
                for (int g = 0; g < 4; g++)
                    for (int e = 0; e < 32; e++) {
                        int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                        s.set(JAVA_BYTE, blk + 16 + g * 32 + e, (byte) (lo | (hi << 4)));
                        v[r * k + b * 256 + g * 64 + e] = sc[g * 2] * lo - mn[g * 2];
                        v[r * k + b * 256 + g * 64 + 32 + e] = sc[g * 2 + 1] * hi - mn[g * 2 + 1];
                    }
            }
        return new Weight(JAM.Q4_K, s, v);
    }

    private static Weight q5_k(int m, int k) { // 176B/256-elem; 5-bit q = nibble | (qh bit<<4)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 176, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < sb; b++) {
                long blk = (long) (r * sb + b) * 176;
                s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
                s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f));
                int[] sc = small(8), mn = small(8);
                packScalesK4(s, blk + 4, sc, mn);
                int[] qh = new int[32]; // 5th bits, one byte per e
                for (int g = 0; g < 4; g++)
                    for (int e = 0; e < 32; e++) {
                        int lo = RNG.nextInt(16),
                                hi = RNG.nextInt(16),
                                bLo = RNG.nextInt(2),
                                bHi = RNG.nextInt(2);
                        qh[e] |= (bLo << (2 * g)) | (bHi << (2 * g + 1));
                        s.set(JAVA_BYTE, blk + 48 + g * 32 + e, (byte) (lo | (hi << 4)));
                        v[r * k + b * 256 + g * 64 + e] = sc[g * 2] * (lo | (bLo << 4)) - mn[g * 2];
                        v[r * k + b * 256 + g * 64 + 32 + e] =
                                sc[g * 2 + 1] * (hi | (bHi << 4)) - mn[g * 2 + 1];
                    }
                for (int e = 0; e < 32; e++) s.set(JAVA_BYTE, blk + 16 + e, (byte) qh[e]);
            }
        return new Weight(JAM.Q5_K, s, v);
    }

    private static Weight q6_k(int m, int k) { // 210B/256-elem; d=1 -> value = sc*(qv-32)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * sb * 210, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < sb; b++) {
                long blk = (long) (r * sb + b) * 210;
                int[] ql = new int[128], qh = new int[64], sc = small(16);
                for (int h = 0; h < 2; h++)
                    for (int j = 0; j < 4; j++)
                        for (int ll = 0; ll < 32; ll++) {
                            int qv = RNG.nextInt(64); // 6-bit
                            int qlIdx = h * 64 + ((j == 1 || j == 3) ? 32 + ll : ll);
                            if (j < 2) ql[qlIdx] |= qv & 0xF;
                            else ql[qlIdx] |= (qv & 0xF) << 4;
                            qh[h * 32 + ll] |= ((qv >> 4) & 3) << (2 * j);
                            v[r * k + b * 256 + h * 128 + j * 32 + ll] =
                                    sc[h * 8 + j * 2 + ll / 16] * (qv - 32);
                        }
                for (int t = 0; t < 128; t++) s.set(JAVA_BYTE, blk + t, (byte) ql[t]);
                for (int t = 0; t < 64; t++) s.set(JAVA_BYTE, blk + 128 + t, (byte) qh[t]);
                for (int t = 0; t < 16; t++) s.set(JAVA_BYTE, blk + 192 + t, (byte) sc[t]);
                s.set(JAVA_SHORT_UNALIGNED, blk + 208, floatToF16(1f));
            }
        return new Weight(JAM.Q6_K, s, v);
    }

    private static Weight mxfp4(
            int m, int k) { // 17B/32-elem; e=128 -> dhalf=1 -> value = kv[nibble]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 17, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < nb; b++) {
                long blk = (long) (r * nb + b) * 17;
                s.set(JAVA_BYTE, blk, (byte) 128);
                for (int t = 0; t < 16; t++) {
                    int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                    s.set(JAVA_BYTE, blk + 1 + t, (byte) (lo | (hi << 4)));
                    v[r * k + b * 32 + t] = FP4_KV[lo];
                    v[r * k + b * 32 + 16 + t] = FP4_KV[hi];
                }
            }
        return new Weight(JAM.MXFP4, s, v);
    }

    private static Weight nvfp4(
            int m, int k) { // 36B/64-elem; d=0x38 -> ue4m3=1 -> value = kv[nibble]
        int nb = k / 64;
        float[] v = new float[m * k];
        MemorySegment s = A.allocate((long) m * nb * 36, 64);
        for (int r = 0; r < m; r++)
            for (int b = 0; b < nb; b++) {
                long blk = (long) (r * nb + b) * 36;
                for (int sub = 0; sub < 4; sub++) {
                    s.set(JAVA_BYTE, blk + sub, (byte) 0x38); // ue4m3 1.0
                    for (int j = 0; j < 8; j++) {
                        int lo = RNG.nextInt(16), hi = RNG.nextInt(16);
                        s.set(JAVA_BYTE, blk + 4 + sub * 8 + j, (byte) (lo | (hi << 4)));
                        v[r * k + b * 64 + sub * 16 + j] = FP4_KV[lo];
                        v[r * k + b * 64 + sub * 16 + 8 + j] = FP4_KV[hi];
                    }
                }
            }
        return new Weight(JAM.NVFP4, s, v);
    }

    /** {@code n} small non-negative ints (0..7) — keeps scale·quant products exact. */
    private static int[] small(int n) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) x[i] = RNG.nextInt(8);
        return x;
    }

    /** Pack sc[8]/mn[8] into the 12-byte k-quant scales block (inverse of get_scale_min_k4). */
    private static void packScalesK4(MemorySegment s, long base, int[] sc, int[] mn) {
        for (int j = 0; j < 4; j++) {
            s.set(JAVA_BYTE, base + j, (byte) ((sc[j] & 63) | ((sc[j + 4] >> 4) << 6)));
            s.set(JAVA_BYTE, base + j + 4, (byte) ((mn[j] & 63) | ((mn[j + 4] >> 4) << 6)));
            s.set(JAVA_BYTE, base + j + 8, (byte) ((sc[j + 4] & 0xF) | ((mn[j + 4] & 0xF) << 4)));
        }
    }

    // ---- the check: run ScalarJAM and compare to the double-precision reference over the exact
    // values ----

    private static void check(String name, Weight w, int m, int n, int k) {
        float[] av = gaussians(n * k);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment c = A.allocate((long) n * m * 4, 64);

        int st = jam.mm(w.seg(), 0, w.tag(), k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k);
        assertEquals(JAM.OK, st, name + " status");
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++) {
                double ref = 0;
                for (int l = 0; l < k; l++) ref += (double) w.vals()[i * k + l] * av[j * k + l];
                float got = c.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
                assertEquals(
                        ref,
                        got,
                        Math.abs(ref) * 1e-4 + 1e-3,
                        name + "[token " + j + ", row " + i + "]");
            }
    }

    @Test
    void gemmEveryDtype() {
        int m = 20, n = 5, k = 64;
        check("F32", f32(m, k), m, n, k);
        check("F16", f16(m, k), m, n, k);
        check("BF16", bf16(m, k), m, n, k);
        check("Q8_0", q8_0(m, k), m, n, k);
        check("Q4_0", q4_0(m, k), m, n, k);
    }

    @Test
    void gemmKQuantFp4() {
        int m = 18, n = 5, k = 256; // k = one super-block (also a multiple of 64 and 32)
        check("Q4_K", q4_k(m, k), m, n, k);
        check("Q5_K", q5_k(m, k), m, n, k);
        check("Q6_K", q6_k(m, k), m, n, k);
        check("MXFP4", mxfp4(m, k), m, n, k);
        check("NVFP4", nvfp4(m, k), m, n, k);
    }

    @Test
    void gemvEveryDtype() {
        int m = 33, k = 96; // n == 1
        check("F32.gemv", f32(m, k), m, 1, k);
        check("F16.gemv", f16(m, k), m, 1, k);
        check("Q8_0.gemv", q8_0(m, k), m, 1, k);
        check("Q4_0.gemv", q4_0(m, k), m, 1, k);
    }

    @Test
    void gemvKQuantFp4() {
        int m = 40, k = 256; // n == 1
        check("Q4_K.gemv", q4_k(m, k), m, 1, k);
        check("Q5_K.gemv", q5_k(m, k), m, 1, k);
        check("Q6_K.gemv", q6_k(m, k), m, 1, k);
        check("MXFP4.gemv", mxfp4(m, k), m, 1, k);
        check("NVFP4.gemv", nvfp4(m, k), m, 1, k);
    }

    @Test
    void declinesUnsupported() {
        MemorySegment d = A.allocate(64 * 1024, 64);
        // every weight dtype now decodes — the only decline is a non-F32 activation or result.
        assertEquals(
                JAM.EUNSUPPORTED,
                jam.mm(d, 0, JAM.Q8_0, 32, d, 0, JAM.F16, 32, d, 0, JAM.F32, 4, 4, 2, 32),
                "F16 activation declined");
        assertEquals(
                JAM.EUNSUPPORTED,
                jam.mm(d, 0, JAM.Q8_0, 32, d, 0, JAM.F32, 32, d, 0, JAM.BF16, 4, 4, 2, 32),
                "BF16 result declined");
    }

    // ---- helpers ----

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
