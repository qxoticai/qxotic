package com.qxotic.jam;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;

/**
 * Shared test fixtures: synthetic quantized weights for every JAM dtype, encoded into a native segment from
 * values the format holds EXACTLY (lossless decode), so a backend's matmul can be checked against a
 * double-precision reference over those same values. Lifted from {@code ScalarJamTest}'s encoders and
 * parameterized by a caller-supplied {@link Random}/{@link Arena} so the cross-backend parity suite can drive
 * every backend through the raw {@link JAM} segment contract — no jinfer, no {@code FloatTensor}.
 */
final class QuantWeights {

    private QuantWeights() {}

    /** A weight: its JAM dtype tag, the encoded native segment, and the exact float values it decodes to. */
    record Weight(int tag, MemorySegment seg, float[] vals) {}

    /** E2M1 FP4 code -> signed magnitude (MXFP4/NVFP4 nibble table). */
    private static final int[] FP4_KV = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

    /** Encode an {@code m×k} weight of the given JAM dtype tag. {@code k} must be a whole number of blocks. */
    static Weight encode(int tag, int m, int k, Arena a, Random rng) {
        return switch (tag) {
            case JAM.F32  -> f32(m, k, a, rng);
            case JAM.F16  -> f16(m, k, a, rng);
            case JAM.BF16 -> bf16(m, k, a, rng);
            case JAM.Q8_0 -> q8_0(m, k, a, rng);
            case JAM.Q4_0 -> q4_0(m, k, a, rng);
            case JAM.Q4_K -> q4_k(m, k, a, rng);
            case JAM.Q5_K -> q5_k(m, k, a, rng);
            case JAM.Q6_K -> q6_k(m, k, a, rng);
            case JAM.MXFP4 -> mxfp4(m, k, a, rng);
            case JAM.NVFP4 -> nvfp4(m, k, a, rng);
            default -> throw new IllegalArgumentException("no encoder for dtype tag " + tag);
        };
    }

    /** A row of {@code n} F32 gaussians in a native segment (an activation/input row). */
    static MemorySegment f32Row(int n, Arena a, Random rng) {
        float[] v = gaussians(n, rng);
        MemorySegment s = a.allocate(v.length * 4L, 64);
        for (int i = 0; i < v.length; i++) s.set(JAVA_FLOAT_UNALIGNED, i * 4L, v[i]);
        return s;
    }

    /**
     * Double-precision reference for weight row {@code i} · activation token {@code j} over {@code k}, as
     * {@code [dot, sumAbs]}. {@code sumAbs = Σ|w·a|} is the error scale for int8-activation backends (native
     * VNNI / register tiles): each product is off by ~the quant step, so the dot's error grows with the sum
     * of magnitudes, not with the (cancellation-prone) result.
     */
    static double[] refDot(Weight w, float[] act, int i, int j, int k) {
        double dot = 0, sumAbs = 0;
        for (int l = 0; l < k; l++) {
            double p = (double) w.vals()[i * k + l] * act[j * k + l];
            dot += p;
            sumAbs += Math.abs(p);
        }
        return new double[]{dot, sumAbs};
    }

    // ---- per-dtype encoders over values the format holds exactly ----

    private static Weight f32(int m, int k, Arena a, Random rng) {
        float[] v = gaussians(m * k, rng);
        MemorySegment s = a.allocate(v.length * 4L, 64);
        for (int i = 0; i < v.length; i++) s.set(JAVA_FLOAT_UNALIGNED, i * 4L, v[i]);
        return new Weight(JAM.F32, s, v);
    }

    private static Weight f16(int m, int k, Arena a, Random rng) {
        float[] v = new float[m * k];
        MemorySegment s = a.allocate(v.length * 2L, 64);
        for (int i = 0; i < v.length; i++) {
            short h = floatToF16((float) rng.nextGaussian());
            s.set(JAVA_SHORT_UNALIGNED, i * 2L, h);
            v[i] = Float.float16ToFloat(h);                 // the exact value f16 holds
        }
        return new Weight(JAM.F16, s, v);
    }

    private static Weight bf16(int m, int k, Arena a, Random rng) {
        float[] v = new float[m * k];
        MemorySegment s = a.allocate(v.length * 2L, 64);
        for (int i = 0; i < v.length; i++) {
            short h = (short) (Float.floatToRawIntBits((float) rng.nextGaussian()) >>> 16);
            s.set(JAVA_SHORT_UNALIGNED, i * 2L, h);
            v[i] = Float.intBitsToFloat((h & 0xFFFF) << 16);
        }
        return new Weight(JAM.BF16, s, v);
    }

    private static Weight q8_0(int m, int k, Arena a, Random rng) {     // scale 1.0, int8 qs -> exact integers
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * nb * 34, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 34;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            for (int e = 0; e < 32; e++) {
                byte q = (byte) (rng.nextInt(256) - 128);   // full int8 range INCLUDING -128 (sign-trick edge)
                s.set(JAVA_BYTE, blk + 2 + e, q);
                v[r * k + b * 32 + e] = q;
            }
        }
        return new Weight(JAM.Q8_0, s, v);
    }

    private static Weight q4_0(int m, int k, Arena a, Random rng) {     // nibble -> value (nibble-8) in [-8,7]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * nb * 18, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 18;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            for (int e = 0; e < 16; e++) {
                int lo = rng.nextInt(16), hi = rng.nextInt(16);
                s.set(JAVA_BYTE, blk + 2 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 32 + e] = lo - 8;
                v[r * k + b * 32 + 16 + e] = hi - 8;
            }
        }
        return new Weight(JAM.Q4_0, s, v);
    }

    private static Weight q4_k(int m, int k, Arena a, Random rng) {     // 144B/256-elem; value = sc*nibble - mn
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * sb * 144, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 144;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));          // d
            s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f));      // dmin
            int[] sc = small(8, rng), mn = small(8, rng);
            packScalesK4(s, blk + 4, sc, mn);
            for (int g = 0; g < 4; g++) for (int e = 0; e < 32; e++) {
                int lo = rng.nextInt(16), hi = rng.nextInt(16);
                s.set(JAVA_BYTE, blk + 16 + g * 32 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 256 + g * 64 + e]      = sc[g * 2]     * lo - mn[g * 2];
                v[r * k + b * 256 + g * 64 + 32 + e] = sc[g * 2 + 1] * hi - mn[g * 2 + 1];
            }
        }
        return new Weight(JAM.Q4_K, s, v);
    }

    private static Weight q5_k(int m, int k, Arena a, Random rng) {     // 176B/256-elem; q = nibble | (qh bit<<4)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * sb * 176, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 176;
            s.set(JAVA_SHORT_UNALIGNED, blk, floatToF16(1f));
            s.set(JAVA_SHORT_UNALIGNED, blk + 2, floatToF16(1f));
            int[] sc = small(8, rng), mn = small(8, rng);
            packScalesK4(s, blk + 4, sc, mn);
            int[] qh = new int[32];                                    // 5th bits, one byte per e
            for (int g = 0; g < 4; g++) for (int e = 0; e < 32; e++) {
                int lo = rng.nextInt(16), hi = rng.nextInt(16), bLo = rng.nextInt(2), bHi = rng.nextInt(2);
                qh[e] |= (bLo << (2 * g)) | (bHi << (2 * g + 1));
                s.set(JAVA_BYTE, blk + 48 + g * 32 + e, (byte) (lo | (hi << 4)));
                v[r * k + b * 256 + g * 64 + e]      = sc[g * 2]     * (lo | (bLo << 4)) - mn[g * 2];
                v[r * k + b * 256 + g * 64 + 32 + e] = sc[g * 2 + 1] * (hi | (bHi << 4)) - mn[g * 2 + 1];
            }
            for (int e = 0; e < 32; e++) s.set(JAVA_BYTE, blk + 16 + e, (byte) qh[e]);
        }
        return new Weight(JAM.Q5_K, s, v);
    }

    private static Weight q6_k(int m, int k, Arena a, Random rng) {     // 210B/256-elem; value = sc*(qv-32)
        int sb = k / 256;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * sb * 210, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < sb; b++) {
            long blk = (long) (r * sb + b) * 210;
            int[] ql = new int[128], qh = new int[64], sc = small(16, rng);
            for (int h = 0; h < 2; h++) for (int j = 0; j < 4; j++) for (int ll = 0; ll < 32; ll++) {
                int qv = rng.nextInt(64);                             // 6-bit
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

    private static Weight mxfp4(int m, int k, Arena a, Random rng) {    // 17B/32-elem; dhalf=1 -> value = kv[nibble]
        int nb = k / 32;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * nb * 17, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 17;
            s.set(JAVA_BYTE, blk, (byte) 128);
            for (int t = 0; t < 16; t++) {
                int lo = rng.nextInt(16), hi = rng.nextInt(16);
                s.set(JAVA_BYTE, blk + 1 + t, (byte) (lo | (hi << 4)));
                v[r * k + b * 32 + t]      = FP4_KV[lo];
                v[r * k + b * 32 + 16 + t] = FP4_KV[hi];
            }
        }
        return new Weight(JAM.MXFP4, s, v);
    }

    private static Weight nvfp4(int m, int k, Arena a, Random rng) {    // 36B/64-elem; ue4m3=1 -> value = kv[nibble]
        int nb = k / 64;
        float[] v = new float[m * k];
        MemorySegment s = a.allocate((long) m * nb * 36, 64);
        for (int r = 0; r < m; r++) for (int b = 0; b < nb; b++) {
            long blk = (long) (r * nb + b) * 36;
            for (int sub = 0; sub < 4; sub++) {
                s.set(JAVA_BYTE, blk + sub, (byte) 0x38);             // ue4m3 1.0
                for (int j = 0; j < 8; j++) {
                    int lo = rng.nextInt(16), hi = rng.nextInt(16);
                    s.set(JAVA_BYTE, blk + 4 + sub * 8 + j, (byte) (lo | (hi << 4)));
                    v[r * k + b * 64 + sub * 16 + j]     = FP4_KV[lo];
                    v[r * k + b * 64 + sub * 16 + 8 + j] = FP4_KV[hi];
                }
            }
        }
        return new Weight(JAM.NVFP4, s, v);
    }

    // ---- helpers ----

    /** {@code n} small non-negative ints (0..7) — keeps scale·quant products exact. */
    private static int[] small(int n, Random rng) {
        int[] x = new int[n];
        for (int i = 0; i < n; i++) x[i] = rng.nextInt(8);
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

    static float[] gaussians(int n, Random rng) {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) v[i] = (float) rng.nextGaussian();
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
