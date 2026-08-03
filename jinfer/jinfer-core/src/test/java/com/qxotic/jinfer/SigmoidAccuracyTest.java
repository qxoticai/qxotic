package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/**
 * {@link Expf#sigmoid} is an approximation and this is its contract: within 4 ulp of the {@code
 * Math.exp}-based reference over the full input range (dense strided sweep of every float region,
 * both signs - the gate logits it serves are unbounded). Saturation must be exact past |x| > 87.3
 * (a hard 1 or 0, matching where the reference rounds to the same). The fused vector path ({@link
 * Expf#sigmoidMulInPlace}) is pinned bit-identical to the scalar mirror.
 */
final class SigmoidAccuracyTest {

    /** The Math.exp-based reference - what the scalar model code computed before. */
    private static float ref(float x) {
        return (float) (1.0 / (1.0 + Math.exp(-x)));
    }

    private static int ulpDiff(float a, float b) {
        return Math.abs(Float.floatToIntBits(a) - Float.floatToIntBits(b));
    }

    @Test
    void sigmoidStaysWithin4UlpOfTheReferenceAcrossBothSigns() {
        // sweep all float bit patterns with |x| <= 88, both signs, prime stride. The SUBNORMAL
        // region is contract-by-absolute-error: below x ~ -87.34 the true sigmoid is a subnormal
        // (<= 1.2e-38) and the polynomial flushes to 0 - the engine's existing subnormal-flush
        // convention (see decodeF16Run), so both being sub-normal-or-zero counts as agreement.
        int worst = 0;
        float worstX = 0;
        int hi = Float.floatToIntBits(88f);
        for (int bits = 0; bits <= hi; bits += 61) {
            for (int sign = 0; sign <= 1; sign++) {
                float x = Float.intBitsToFloat(sign == 0 ? bits : bits | 0x80000000);
                float got = Expf.sigmoid(x);
                float r = ref(x);
                if (Math.abs(r) < Float.MIN_NORMAL && Math.abs(got) < Float.MIN_NORMAL) {
                    continue; // both subnormal-or-zero: absolute error <= 1.2e-38
                }
                int u = ulpDiff(got, r);
                if (u > worst) {
                    worst = u;
                    worstX = x;
                }
            }
        }
        assertTrue(
                worst <= 4,
                "sigmoid degraded to "
                        + worst
                        + " ulp at x="
                        + worstX
                        + " (got "
                        + Expf.sigmoid(worstX)
                        + ", ref "
                        + ref(worstX)
                        + ")");
    }

    @Test
    void saturationAndCenterAreExact() {
        assertEquals(0.5f, Expf.sigmoid(0f), "sigmoid(0) must be exactly 0.5");
        assertEquals(0.5f, Expf.sigmoid(-0f));
        assertEquals(1f, Expf.sigmoid(88f), "positive saturation");
        assertEquals(0f, Expf.sigmoid(-88f), "negative saturation");
        assertEquals(1f, Expf.sigmoid(Float.POSITIVE_INFINITY));
        assertEquals(0f, Expf.sigmoid(Float.NEGATIVE_INFINITY));
    }

    @Test
    void vectorGateMatchesTheScalarMirrorBitForBit() {
        java.util.Random rnd = new java.util.Random(7);
        int n = 300; // odd tail on purpose
        try (Arena arena = Arena.ofConfined()) {
            F32FloatTensor out = F32FloatTensor.allocate(arena, n);
            float[] gate = new float[n];
            float[] initial = new float[n];
            for (int trial = 0; trial < 100; trial++) {
                for (int i = 0; i < n; i++) {
                    gate[i] = (rnd.nextFloat() - 0.5f) * (trial % 4 == 0 ? 200f : 12f);
                    initial[i] = rnd.nextFloat() * 4f - 2f;
                    out.setFloat(i, initial[i]);
                }
                Expf.sigmoidMulInPlace(out, 0, gate, 0, n);
                for (int i = 0; i < n; i++) {
                    float expected = initial[i] * Expf.sigmoid(gate[i]);
                    assertEquals(
                            Float.floatToIntBits(expected),
                            Float.floatToIntBits(out.getFloat(i)),
                            "vector gate diverged from the scalar mirror at " + i);
                }
            }
        }
    }
}
