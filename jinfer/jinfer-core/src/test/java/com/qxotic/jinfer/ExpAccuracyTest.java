package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * The polynomial exp behind the fused softmax pass is an APPROXIMATION, and this is its contract:
 * over the entire domain the kernel uses (softmax arguments, {@code [-87.33654, 0]}) it must stay
 * within 3 ulp of {@code (float) Math.exp} - the exhaustively-measured bound at introduction
 * (2026-08-03: all 1.12e9 domain floats swept; 98.61% bit-exact, worst 3 ulp, max relative error
 * 2.5e-7, versus the ~5e-4 rounding already present in the F16 KV rows that feed the scores). A
 * coefficient or range-reduction edit that degrades the approximation fails here, not in a model's
 * reply.
 *
 * <p>The sweep here strides the domain's bit patterns with a prime step (~18M evaluations, &lt;1s)
 * plus the exact boundary cases; the vector lane path is pinned BIT-IDENTICAL to the scalar mirror
 * (same fma sequence by construction - only reduction order may ever differ, and exp has none).
 */
final class ExpAccuracyTest {

    private static final float DOMAIN_LO = -87.33654f; // FlashAttention.EXP_UNDERFLOW

    private static int ulpDiff(float a, float b) {
        return Math.abs(Float.floatToIntBits(a) - Float.floatToIntBits(b));
    }

    @Test
    void polynomialStaysWithin3UlpOfMathExpAcrossTheDomain() {
        int loBits = Float.floatToIntBits(DOMAIN_LO);
        int hiBits = Float.floatToIntBits(-Float.MIN_VALUE);
        int worst = 0;
        float worstX = 0;
        long n = 0;
        for (int bits = hiBits; bits <= loBits; bits += 61) { // prime stride: ~18M points
            float x = Float.intBitsToFloat(bits);
            int u = ulpDiff(FastMath.expNeg(x), (float) Math.exp(x));
            if (u > worst) {
                worst = u;
                worstX = x;
            }
            n++;
        }
        assertTrue(
                worst <= 3,
                "polynomial exp degraded to "
                        + worst
                        + " ulp at x="
                        + worstX
                        + " (contract: <= 3 ulp vs (float) Math.exp) over "
                        + n
                        + " points");
    }

    @Test
    void boundariesAreExact() {
        assertEquals(1f, FastMath.expNeg(0f), "exp(0) must be exactly 1");
        assertEquals(1f, FastMath.expNeg(-0f));
        assertEquals(0f, FastMath.expNeg(Float.NEGATIVE_INFINITY), "masked scores");
        assertEquals(
                0f,
                FastMath.expNeg(Math.nextDown(DOMAIN_LO)),
                "below the underflow cutoff the pass returns exactly 0 (true value <= 1.2e-38)");
        assertTrue(FastMath.expNeg(DOMAIN_LO) > 0f, "the cutoff itself still evaluates");
    }

    @Test
    void vectorLanesMatchTheScalarMirrorBitForBit() {
        java.util.Random rnd = new java.util.Random(42);
        float[] row = new float[256];
        float[] expected = new float[row.length];
        for (int trial = 0; trial < 200; trial++) {
            float max = trial == 0 ? 0f : rnd.nextFloat() * 20f;
            for (int i = 0; i < row.length; i++) {
                // scores below, at, and far below the row max - incl. the underflow region
                row[i] = max - (trial % 3 == 0 ? rnd.nextFloat() * 100f : rnd.nextFloat() * 8f);
            }
            for (int i = 0; i < row.length; i++) {
                expected[i] = FastMath.expNeg(row[i] - max);
            }
            double sum = FlashAttention.expRowInPlace(row, 0, row.length, max);
            double expectedSum = 0;
            for (int i = 0; i < row.length; i++) {
                assertEquals(
                        Float.floatToIntBits(expected[i]),
                        Float.floatToIntBits(row[i]),
                        "vector lane diverged from the scalar mirror at " + i);
                expectedSum += expected[i];
            }
            // reduction order differs (lanes then across); allow only that
            assertTrue(
                    Math.abs(sum - expectedSum) <= Math.ulp((float) expectedSum) * row.length,
                    "sum drifted beyond reduction-order noise");
        }
    }
}
