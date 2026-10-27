package com.qxotic.jinfer.models.lfm3;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;

/**
 * The exact GELU, {@code 0.5 x (1 + erf(x / sqrt 2))} - HF's {@code "gelu"}, ggml's {@code
 * gelu_erf} - of both LFM3 projectors. Java has no {@code erf}; this one is double precision, so
 * the float result agrees with libm's {@code erff} to the last bit or so.
 */
final class GeluErf {

    private static final double TWO_OVER_SQRT_PI = 2 / Math.sqrt(Math.PI);
    private static final float SQRT_2_INV = (float) (1 / Math.sqrt(2));

    private GeluErf() {}

    /** In place over {@code rows} rows of {@code width}, rows in parallel. */
    static void rows(MemoryView<MemorySegment> t, int rows, int width) {
        Parallel.forLoop(
                rows,
                r ->
                        Ops.mapInPlace(
                                t,
                                (long) r * width,
                                width,
                                x -> 0.5f * x * (1f + erf(x * SQRT_2_INV))));
    }

    /**
     * {@code erf(x) = 2/sqrt(pi) e^(-x^2) sum_n (2x^2)^n x / (1 3 5 ... (2n+1))}: every term
     * positive, so no cancellation; beyond |x| = 4 erf is 1 in float.
     */
    static float erf(float x) {
        double a = Math.abs(x);
        if (a >= 4) return Math.copySign(1f, x);
        double term = a, sum = a, twoA2 = 2 * a * a;
        for (int n = 1; term > 1e-17 * sum; n++) {
            term *= twoA2 / (2 * n + 1);
            sum += term;
        }
        return (float) Math.copySign(TWO_OVER_SQRT_PI * Math.exp(-a * a) * sum, x);
    }
}
