package com.qxotic.jinfer.x.kernels;

/**
 * Fast float transcendentals for the hot paths: e^x as 2^n * e^r (n = round(x*log2e) via the
 * magic-number trick, r = x - n*ln2 hi/lo split, e^r a degree-6 Horner polynomial), and sigmoid
 * built on it. {@code Math.exp} costs ~9ns per call in a native image (no vector math stubs, no
 * SVML); the fused vector form of this polynomial runs at ~0.19ns per element.
 *
 * <p>Ported byte-for-byte from jinfer-core {@code FastMath}. The segment-coupled fused vector forms
 * ({@code sigmoidMulInPlace}, {@code tanhInPlace}, {@code expSumInPlace}) are deferred until a
 * ported consumer needs them — their accuracy gates live with the consumers.
 *
 * <p>These are APPROXIMATIONS with enforced contracts: every consumer domain carries an exhaustive
 * or dense-sweep accuracy gate ({@code ExpAccuracyTest}: exp within 3 ulp of {@code (float)
 * Math.exp} over the whole softmax domain [-87.33654, 0]; {@code SigmoidAccuracyTest} for sigmoid).
 * A new use site with a new input domain gets its own gate before it ships.
 */
public final class FastMath {

    private FastMath() {}

    static final float EXP_LOG2E = 1.4426950408889634f;
    static final float EXP_MAGIC = 12582912.0f; // 1.5*2^23: fma with it rounds to nearest
    static final float EXP_NLN2_HI = -0.6931471824645996f;
    static final float EXP_NLN2_LO = 1.904654323148236e-9f;
    static final float EXP_UNDERFLOW = -87.33654f; // float e^x is 0-or-subnormal below
    static final float EXP_C6 = 1f / 720f,
            EXP_C5 = 1f / 120f,
            EXP_C4 = 1f / 24f,
            EXP_C3 = 1f / 6f,
            EXP_C2 = 0.5f;

    /**
     * e^x for {@code x <= 0} (the softmax domain; the 2^n splice overflows past x ~ +88). Below
     * {@link #EXP_UNDERFLOW} returns exactly 0 (the true value is a subnormal {@code <= 1.2e-38}).
     */
    public static float expNeg(float x) {
        if (x < EXP_UNDERFLOW) return 0f;
        float t = Math.fma(x, EXP_LOG2E, EXP_MAGIC);
        float n = t - EXP_MAGIC;
        float r = Math.fma(n, EXP_NLN2_HI, x);
        r = Math.fma(n, EXP_NLN2_LO, r);
        float twoN = Float.intBitsToFloat(((int) n + 127) << 23);
        float p = EXP_C6;
        p = Math.fma(p, r, EXP_C5);
        p = Math.fma(p, r, EXP_C4);
        p = Math.fma(p, r, EXP_C3);
        p = Math.fma(p, r, EXP_C2);
        p = Math.fma(p, r, 1f);
        p = Math.fma(p, r, 1f);
        return p * twoN;
    }

    /**
     * Sigmoid over the full float range, via {@code e = expNeg(-|x|)}: for {@code x >= 0}, {@code
     * 1/(1+e)}; for {@code x < 0}, {@code e/(1+e)} - computed as that DIVISION, never as {@code 1 -
     * 1/(1+e)}, which cancels catastrophically once e drops below the ulp of 1 ({@code x < -15.9}
     * would return 0 instead of e; caught by SigmoidAccuracyTest at introduction). Saturates to
     * exactly 1 past {@code x > 87.3} and to exactly 0 past {@code x < -87.34}, where the true
     * value is a subnormal ({@code <= 1.2e-38}) - flushed, matching the engine's existing subnormal
     * convention.
     */
    public static float sigmoid(float x) {
        // branch-free: -|x| by forcing the sign bit, numerator selected by sign-mask bit blend.
        // The ternary form costs 2.6x on random-signed gates (measured: 4.26 vs 1.61 ns/op -
        // 50/50 signs shred the branch predictor); this is also bit-identical to the vector body.
        int bits = Float.floatToRawIntBits(x);
        float e = expNeg(Float.intBitsToFloat(bits | Integer.MIN_VALUE));
        int m = bits >> 31; // all-ones iff negative
        float num = Float.intBitsToFloat((0x3F800000 & ~m) | (Float.floatToRawIntBits(e) & m));
        return num / (1f + e);
    }

    /**
     * tanh over the full float range: {@code sign(x) * (1-t)/(1+t)} with {@code t = expNeg(-2|x|)},
     * all sign handling via bit ops (branch-free). Saturates to exactly +-1 past |x| > 43.7
     * (agreeing with the reference well before that - float tanh rounds to 1 past ~8.7).
     */
    public static float tanh(float x) {
        int bits = Float.floatToRawIntBits(x);
        int sign = bits & 0x80000000;
        float m2ax = Float.intBitsToFloat(bits & 0x7FFFFFFF) * -2f; // -2|x| <= 0
        float t = expNeg(m2ax);
        float r = (1f - t) / (1f + t);
        return Float.intBitsToFloat(Float.floatToRawIntBits(r) | sign);
    }
}
