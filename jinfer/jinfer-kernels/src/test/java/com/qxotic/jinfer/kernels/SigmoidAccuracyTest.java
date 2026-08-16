package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.junit.jupiter.api.Test;

/**
 * {@link FastMath#sigmoid} is an approximation and this is its contract: within 4 ulp of the {@code
 * Math.exp}-based reference over the full input range (dense strided sweep of every float region,
 * both signs - the gate logits it serves are unbounded). Saturation must be exact past |x| > 87.3
 * (a hard 1 or 0, matching where the reference rounds to the same). The fused vector form ({@code
 * sigmoidMulInPlace}) is deferred until a ported consumer needs it - its bit-parity gate ships with
 * that consumer.
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
                float got = FastMath.sigmoid(x);
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
                        + FastMath.sigmoid(worstX)
                        + ", ref "
                        + ref(worstX)
                        + ")");
    }

    @Test
    void saturationAndCenterAreExact() {
        assertEquals(0.5f, FastMath.sigmoid(0f), "sigmoid(0) must be exactly 0.5");
        assertEquals(0.5f, FastMath.sigmoid(-0f));
        assertEquals(1f, FastMath.sigmoid(88f), "positive saturation");
        assertEquals(0f, FastMath.sigmoid(-88f), "negative saturation");
        assertEquals(1f, FastMath.sigmoid(Float.POSITIVE_INFINITY));
        assertEquals(0f, FastMath.sigmoid(Float.NEGATIVE_INFINITY));
    }
}
