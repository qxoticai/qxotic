package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/**
 * {@link FastMath#tanh}'s contract, in two regimes: for |x| >= 0.25 within 8 ulp of {@code (float)
 * Math.tanh}; below that the {@code (1-t)} subtraction bounds the ABSOLUTE error at the ulp of 1
 * (~1.2e-7) rather than the relative error - measured and acceptable for its consumer, the Inflect2
 * waveform soft-clip (24-bit audio's LSB is 6e-8, and near zero tanh(x) ~ x so the signal itself
 * dwarfs the error). Odd symmetry and saturation are exact. The fused vector path is pinned
 * bit-identical to the scalar mirror.
 */
final class TanhAccuracyTest {

    private static int ulpDiff(float a, float b) {
        return Math.abs(Float.floatToIntBits(a) - Float.floatToIntBits(b));
    }

    @Test
    void tanhStaysWithinContractAcrossBothSigns() {
        int worstUlp = 0;
        float worstUlpX = 0;
        double worstAbs = 0;
        float worstAbsX = 0;
        int lo = Float.floatToIntBits(0.25f);
        int hi = Float.floatToIntBits(50f);
        for (int bits = lo; bits <= hi; bits += 61) { // |x| in [0.25, 50]: ulp regime
            for (int sign = 0; sign <= 1; sign++) {
                float x = Float.intBitsToFloat(sign == 0 ? bits : bits | 0x80000000);
                int u = ulpDiff(FastMath.tanh(x), (float) Math.tanh(x));
                if (u > worstUlp) {
                    worstUlp = u;
                    worstUlpX = x;
                }
            }
        }
        for (int bits = 0; bits < lo; bits += 61) { // |x| < 0.25: absolute regime
            for (int sign = 0; sign <= 1; sign++) {
                float x = Float.intBitsToFloat(sign == 0 ? bits : bits | 0x80000000);
                double abs = Math.abs((double) FastMath.tanh(x) - (float) Math.tanh(x));
                if (abs > worstAbs) {
                    worstAbs = abs;
                    worstAbsX = x;
                }
            }
        }
        assertTrue(
                worstUlp <= 8,
                "tanh degraded to " + worstUlp + " ulp at x=" + worstUlpX + " (contract: <= 8)");
        assertTrue(
                worstAbs <= 1.2e-7,
                "tanh absolute error "
                        + worstAbs
                        + " at x="
                        + worstAbsX
                        + " (contract: <= 1.2e-7)");
    }

    @Test
    void symmetryAndSaturationAreExact() {
        assertEquals(0f, FastMath.tanh(0f));
        assertEquals(
                Float.floatToIntBits(-0f), Float.floatToIntBits(FastMath.tanh(-0f)), "odd at -0");
        assertEquals(1f, FastMath.tanh(50f), "positive saturation");
        assertEquals(-1f, FastMath.tanh(-50f), "negative saturation");
        assertEquals(1f, FastMath.tanh(Float.POSITIVE_INFINITY));
        assertEquals(-1f, FastMath.tanh(Float.NEGATIVE_INFINITY));
        java.util.Random rnd = new java.util.Random(11);
        for (int i = 0; i < 100_000; i++) {
            float x = (rnd.nextFloat() - 0.5f) * 20f;
            assertEquals(
                    Float.floatToIntBits(-FastMath.tanh(x)),
                    Float.floatToIntBits(FastMath.tanh(-x)),
                    "odd symmetry must be exact (sign-bit construction) at " + x);
        }
    }

    @Test
    void vectorPassMatchesTheScalarMirrorBitForBit() {
        java.util.Random rnd = new java.util.Random(23);
        int n = 300; // odd tail on purpose
        try (Arena arena = Arena.ofConfined()) {
            F32FloatTensor t = F32FloatTensor.allocate(arena, n);
            float[] input = new float[n];
            for (int trial = 0; trial < 100; trial++) {
                for (int i = 0; i < n; i++) {
                    input[i] = (rnd.nextFloat() - 0.5f) * (trial % 4 == 0 ? 100f : 6f);
                    t.setFloat(i, input[i]);
                }
                FastMath.tanhInPlace(t, 0, n);
                for (int i = 0; i < n; i++) {
                    assertEquals(
                            Float.floatToIntBits(FastMath.tanh(input[i])),
                            Float.floatToIntBits(t.getFloat(i)),
                            "vector tanh diverged from the scalar mirror at " + i);
                }
            }
        }
    }
}
