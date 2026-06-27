package com.qxotic.jam;

import org.junit.jupiter.api.Test;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;
import static java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED;
import static java.lang.foreign.ValueLayout.JAVA_SHORT_UNALIGNED;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * {@link ScalarJAM} correctness. Each implemented dtype is encoded from values it represents EXACTLY (so the
 * decode is lossless), and the matmul must equal a double-precision reference computed from those same values.
 * Covers gemm + gemv across the dtypes, strided operands, and the decline contract. Self-contained in
 * jam-scalar (no jinfer, no native).
 */
class ScalarJamTest {

    private static final JAM jam = new ScalarJAM();
    private static final Arena A = Arena.ofAuto();
    private static final Random RNG = new Random(7);

    /** A weight: its dtype tag, the encoded native segment, and the exact float values it decodes to. */
    private record Weight(int tag, MemorySegment seg, float[] vals) {}

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
            v[i] = ScalarJAM.f16ToFloat(h);                 // the exact value f16 holds
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

    // ---- the check: run ScalarJAM and compare to the double-precision reference over the exact values ----

    private static void check(String name, Weight w, int m, int n, int k) {
        float[] av = gaussians(n * k);
        MemorySegment a = A.allocate(av.length * 4L, 64);
        for (int i = 0; i < av.length; i++) a.set(JAVA_FLOAT_UNALIGNED, i * 4L, av[i]);
        MemorySegment c = A.allocate((long) n * m * 4, 64);

        int st = jam.mm(w.seg(), 0, w.tag(), k, a, 0, JAM.F32, k, c, 0, JAM.F32, m, m, n, k);
        assertEquals(JAM.OK, st, name + " status");
        for (int j = 0; j < n; j++) for (int i = 0; i < m; i++) {
            double ref = 0;
            for (int l = 0; l < k; l++) ref += (double) w.vals()[i * k + l] * av[j * k + l];
            float got = c.get(JAVA_FLOAT_UNALIGNED, ((long) j * m + i) * 4);
            assertEquals(ref, got, Math.abs(ref) * 1e-4 + 1e-3, name + "[token " + j + ", row " + i + "]");
        }
    }

    @Test void gemmEveryDtype() {
        int m = 20, n = 5, k = 64;
        check("F32",  f32(m, k),  m, n, k);
        check("F16",  f16(m, k),  m, n, k);
        check("BF16", bf16(m, k), m, n, k);
        check("Q8_0", q8_0(m, k), m, n, k);
        check("Q4_0", q4_0(m, k), m, n, k);
    }

    @Test void gemvEveryDtype() {
        int m = 33, k = 96;   // n == 1
        check("F32.gemv",  f32(m, k),  m, 1, k);
        check("F16.gemv",  f16(m, k),  m, 1, k);
        check("Q8_0.gemv", q8_0(m, k), m, 1, k);
        check("Q4_0.gemv", q4_0(m, k), m, 1, k);
    }

    @Test void declinesUnsupported() {
        MemorySegment d = A.allocate(64 * 1024, 64);
        // k-quants/fp4 not yet decoded
        assertEquals(JAM.EUNSUPPORTED, jam.mm(d, 0, JAM.Q4_K, 256, d, 0, JAM.F32, 256, d, 0, JAM.F32, 4, 4, 2, 256), "Q4_K declined");
        assertEquals(JAM.EUNSUPPORTED, jam.mm(d, 0, JAM.MXFP4, 32, d, 0, JAM.F32, 32, d, 0, JAM.F32, 4, 4, 2, 32), "MXFP4 declined");
        // non-F32 activation / result
        assertEquals(JAM.EUNSUPPORTED, jam.mm(d, 0, JAM.Q8_0, 32, d, 0, JAM.F16, 32, d, 0, JAM.F32, 4, 4, 2, 32), "F16 activation declined");
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
