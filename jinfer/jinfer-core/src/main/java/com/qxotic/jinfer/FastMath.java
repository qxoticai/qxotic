package com.qxotic.jinfer;

import java.lang.foreign.MemorySegment;
import java.nio.ByteOrder;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;

/**
 * Fast float transcendentals for the hot paths: e^x as 2^n * e^r (n = round(x*log2e) via the
 * magic-number trick, r = x - n*ln2 hi/lo split, e^r a degree-6 Horner polynomial), and sigmoid
 * built on it. {@code Math.exp} costs ~9ns per call in a native image (no vector math stubs, no
 * SVML); the fused vector form of this polynomial runs at ~0.19ns per element.
 *
 * <p>These are APPROXIMATIONS with enforced contracts: every consumer domain carries an exhaustive
 * or dense-sweep accuracy gate ({@code ExpAccuracyTest}: exp within 3 ulp of {@code (float)
 * Math.exp} over the whole softmax domain [-87.33654, 0]; {@code SigmoidAccuracyTest} for sigmoid).
 * A new use site with a new input domain gets its own gate before it ships.
 *
 * <p>The scalar methods here are the MIRRORS - identical fma sequences to the vector bodies, used
 * for loop tails and non-vector fallbacks so both paths agree bit for bit lane-wise. The vector
 * bodies themselves are fused INLINE at each use site ({@code FlashAttention.expRowInPlace}, {@code
 * F32FloatTensor.softmaxInPlace}, {@link #sigmoidMulInPlace}): routed through a helper - even
 * {@code @AlwaysInline} - the native-image Vector API expansion phase leaves them boxed at scalar
 * speed (measured, same trap as {@code pvTile}).
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
     * {@code out[i] *= sigmoid(gate[i])} over {@code n} elements - the gated-attention output
     * scaling (Qwen3.5), fused and vectorized. The vector body mirrors {@link #sigmoid} exactly.
     */
    public static void sigmoidMulInPlace(
            F32FloatTensor out, long outOffset, float[] gate, int gateOffset, int n) {
        int i = 0;
        if (FloatTensor.USE_VECTOR_API) {
            var sp = FloatTensor.F_SPECIES;
            int len = sp.length();
            int bound = sp.loopBound(n);
            if (bound > 0) {
                FloatVector vLog2e = FloatVector.broadcast(sp, EXP_LOG2E);
                FloatVector vMagic = FloatVector.broadcast(sp, EXP_MAGIC);
                FloatVector vHi = FloatVector.broadcast(sp, EXP_NLN2_HI);
                FloatVector vLo = FloatVector.broadcast(sp, EXP_NLN2_LO);
                FloatVector vC6 = FloatVector.broadcast(sp, EXP_C6);
                FloatVector vC5 = FloatVector.broadcast(sp, EXP_C5);
                FloatVector vC4 = FloatVector.broadcast(sp, EXP_C4);
                FloatVector vC3 = FloatVector.broadcast(sp, EXP_C3);
                FloatVector vC2 = FloatVector.broadcast(sp, EXP_C2);
                FloatVector vOne = FloatVector.broadcast(sp, 1f);
                FloatVector vZero = FloatVector.zero(sp);
                FloatVector vUnder = FloatVector.broadcast(sp, EXP_UNDERFLOW);
                for (; i < bound; i += len) {
                    FloatVector x = FloatVector.fromArray(sp, gate, gateOffset + i);
                    // -|x| <= 0 (expNeg's argument) by forcing the sign bit - one op, and the
                    // exact mirror of the scalar path's bit trick
                    FloatVector xn =
                            x.reinterpretAsInts().or(Integer.MIN_VALUE).reinterpretAsFloats();
                    FloatVector xc = xn.max(vUnder);
                    FloatVector t = xc.fma(vLog2e, vMagic);
                    FloatVector nn = t.sub(vMagic);
                    FloatVector r = nn.fma(vHi, xc);
                    r = nn.fma(vLo, r);
                    IntVector eb =
                            ((IntVector) nn.convert(VectorOperators.F2I, 0))
                                    .add(127)
                                    .lanewise(VectorOperators.LSHL, 23);
                    FloatVector p = vC6.fma(r, vC5);
                    p = p.fma(r, vC4);
                    p = p.fma(r, vC3);
                    p = p.fma(r, vC2);
                    p = p.fma(r, vOne);
                    p = p.fma(r, vOne);
                    FloatVector e =
                            p.mul(eb.reinterpretAsFloats())
                                    .blend(vZero, xn.compare(VectorOperators.LT, vUnder));
                    // x >= 0: 1/(1+e); x < 0: e/(1+e) as a DIRECT division (1 - 1/(1+e) cancels
                    // catastrophically below the ulp of 1) - one div via a blended numerator
                    FloatVector num = vOne.blend(e, x.compare(VectorOperators.LT, vZero));
                    FloatVector sig = num.div(e.add(vOne));
                    long byteOffset = (long) (outOffset + i) * Float.BYTES;
                    FloatVector.fromMemorySegment(
                                    sp, out.vseg, out.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN)
                            .mul(sig)
                            .intoMemorySegment(
                                    out.vseg, out.vbase + byteOffset, ByteOrder.LITTLE_ENDIAN);
                }
            }
        }
        for (; i < n; i++) {
            out.setFloat(
                    outOffset + i, out.getFloat(outOffset + i) * sigmoid(gate[gateOffset + i]));
        }
    }

    /**
     * tanh over the full float range: {@code sign(x) * (1-t)/(1+t)} with {@code t = expNeg(-2|x|)},
     * all sign handling via bit ops (branch-free). Saturates to exactly +-1 past |x| > 43.7
     * (agreeing with the reference well before that - float tanh rounds to 1 past ~8.7). Near zero
     * the (1-t) subtraction bounds the ABSOLUTE error at ~6e-8 (the ulp of 1) rather than the
     * relative error - fine for its consumer, the Inflect2 waveform soft-clip (24-bit audio's LSB);
     * see TanhAccuracyTest for the enforced contract.
     */
    public static float tanh(float x) {
        int bits = Float.floatToRawIntBits(x);
        int sign = bits & 0x80000000;
        float m2ax = Float.intBitsToFloat(bits & 0x7FFFFFFF) * -2f; // -2|x| <= 0
        float t = expNeg(m2ax);
        float r = (1f - t) / (1f + t);
        return Float.intBitsToFloat(Float.floatToRawIntBits(r) | sign);
    }

    /** {@code t[i] = tanh(t[i])} over {@code n} elements, fused - the waveform soft-clip. */
    public static void tanhInPlace(F32FloatTensor t, long offset, int n) {
        int i = 0;
        if (FloatTensor.USE_VECTOR_API) {
            var sp = FloatTensor.F_SPECIES;
            int len = sp.length();
            int bound = sp.loopBound(n);
            if (bound > 0) {
                FloatVector vLog2e = FloatVector.broadcast(sp, EXP_LOG2E);
                FloatVector vMagic = FloatVector.broadcast(sp, EXP_MAGIC);
                FloatVector vHi = FloatVector.broadcast(sp, EXP_NLN2_HI);
                FloatVector vLo = FloatVector.broadcast(sp, EXP_NLN2_LO);
                FloatVector vC6 = FloatVector.broadcast(sp, EXP_C6);
                FloatVector vC5 = FloatVector.broadcast(sp, EXP_C5);
                FloatVector vC4 = FloatVector.broadcast(sp, EXP_C4);
                FloatVector vC3 = FloatVector.broadcast(sp, EXP_C3);
                FloatVector vC2 = FloatVector.broadcast(sp, EXP_C2);
                FloatVector vOne = FloatVector.broadcast(sp, 1f);
                FloatVector vZero = FloatVector.zero(sp);
                FloatVector vUnder = FloatVector.broadcast(sp, EXP_UNDERFLOW);
                MemorySegment seg = t.vseg;
                for (; i < bound; i += len) {
                    long byteOffset = t.vbase + (long) (offset + i) * Float.BYTES;
                    FloatVector x =
                            FloatVector.fromMemorySegment(
                                    sp, seg, byteOffset, ByteOrder.LITTLE_ENDIAN);
                    IntVector xb = x.reinterpretAsInts();
                    IntVector sign = xb.and(Integer.MIN_VALUE);
                    FloatVector m2ax =
                            xb.and(Integer.MAX_VALUE).reinterpretAsFloats().mul(-2f); // -2|x|
                    FloatVector xc = m2ax.max(vUnder);
                    FloatVector tt = xc.fma(vLog2e, vMagic);
                    FloatVector nn = tt.sub(vMagic);
                    FloatVector r = nn.fma(vHi, xc);
                    r = nn.fma(vLo, r);
                    IntVector eb =
                            ((IntVector) nn.convert(VectorOperators.F2I, 0))
                                    .add(127)
                                    .lanewise(VectorOperators.LSHL, 23);
                    FloatVector p = vC6.fma(r, vC5);
                    p = p.fma(r, vC4);
                    p = p.fma(r, vC3);
                    p = p.fma(r, vC2);
                    p = p.fma(r, vOne);
                    p = p.fma(r, vOne);
                    FloatVector e =
                            p.mul(eb.reinterpretAsFloats())
                                    .blend(vZero, m2ax.compare(VectorOperators.LT, vUnder));
                    FloatVector th = vOne.sub(e).div(vOne.add(e));
                    th.reinterpretAsInts()
                            .or(sign)
                            .reinterpretAsFloats()
                            .intoMemorySegment(seg, byteOffset, ByteOrder.LITTLE_ENDIAN);
                }
            }
        }
        for (; i < n; i++) {
            t.setFloat(offset + i, tanh(t.getFloat(offset + i)));
        }
    }

    /**
     * The fused exp+sum leg of a softmax over an F32 span: {@code t[i] = e^(t[i]-max)} in place,
     * returning the sum - the vector mirror of {@link #expNeg} over a memory segment. Used by
     * {@link F32FloatTensor#softmaxInPlace} (the sampler's vocab softmax and the MoE routers).
     */
    static double expSumInPlace(F32FloatTensor t, long offset, int n, float max) {
        return expSum(t, offset, n, max, true);
    }

    /** As {@link #expSumInPlace} but read-only: the sum without writing the exponentials back. */
    static double expSum(F32FloatTensor t, long offset, int n, float max) {
        return expSum(t, offset, n, max, false);
    }

    private static double expSum(F32FloatTensor t, long offset, int n, float max, boolean store) {
        int i = 0;
        double sum = 0;
        if (FloatTensor.USE_VECTOR_API) {
            var sp = FloatTensor.F_SPECIES;
            int len = sp.length();
            int bound = sp.loopBound(n);
            if (bound > 0) {
                FloatVector mv = FloatVector.broadcast(sp, max);
                FloatVector acc = FloatVector.zero(sp);
                FloatVector vLog2e = FloatVector.broadcast(sp, EXP_LOG2E);
                FloatVector vMagic = FloatVector.broadcast(sp, EXP_MAGIC);
                FloatVector vHi = FloatVector.broadcast(sp, EXP_NLN2_HI);
                FloatVector vLo = FloatVector.broadcast(sp, EXP_NLN2_LO);
                FloatVector vC6 = FloatVector.broadcast(sp, EXP_C6);
                FloatVector vC5 = FloatVector.broadcast(sp, EXP_C5);
                FloatVector vC4 = FloatVector.broadcast(sp, EXP_C4);
                FloatVector vC3 = FloatVector.broadcast(sp, EXP_C3);
                FloatVector vC2 = FloatVector.broadcast(sp, EXP_C2);
                FloatVector vOne = FloatVector.broadcast(sp, 1f);
                FloatVector vZero = FloatVector.zero(sp);
                FloatVector vUnder = FloatVector.broadcast(sp, EXP_UNDERFLOW);
                MemorySegment seg = t.vseg;
                for (; i < bound; i += len) {
                    long byteOffset = t.vbase + (long) (offset + i) * Float.BYTES;
                    FloatVector x =
                            FloatVector.fromMemorySegment(
                                            sp, seg, byteOffset, ByteOrder.LITTLE_ENDIAN)
                                    .sub(mv);
                    FloatVector xc = x.max(vUnder);
                    FloatVector tt = xc.fma(vLog2e, vMagic);
                    FloatVector nn = tt.sub(vMagic);
                    FloatVector r = nn.fma(vHi, xc);
                    r = nn.fma(vLo, r);
                    IntVector eb =
                            ((IntVector) nn.convert(VectorOperators.F2I, 0))
                                    .add(127)
                                    .lanewise(VectorOperators.LSHL, 23);
                    FloatVector p = vC6.fma(r, vC5);
                    p = p.fma(r, vC4);
                    p = p.fma(r, vC3);
                    p = p.fma(r, vC2);
                    p = p.fma(r, vOne);
                    p = p.fma(r, vOne);
                    p =
                            p.mul(eb.reinterpretAsFloats())
                                    .blend(vZero, x.compare(VectorOperators.LT, vUnder));
                    if (store) p.intoMemorySegment(seg, byteOffset, ByteOrder.LITTLE_ENDIAN);
                    acc = acc.add(p);
                }
                sum = acc.reduceLanes(VectorOperators.ADD);
            }
        }
        for (; i < n; i++) {
            float p = expNeg(t.getFloat(offset + i) - max);
            if (store) t.setFloat(offset + i, p);
            sum += p;
        }
        return sum;
    }
}
