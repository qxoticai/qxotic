package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;

/**
 * {@code reduce_precision}: rounds each float to {@code exponentBits} / {@code mantissaBits} as XLA
 * does ({@code EmitReducePrecisionIR}): the mantissa is rounded to nearest even, then values whose
 * exponent overflows become infinities and those that underflow (including the reduced format's
 * subnormals) become signed zeros; NaNs stay NaNs while the reduced format keeps a mantissa.
 */
final class Precision {

    private Precision() {}

    static Tensor reduce(Tensor x, int exponentBits, int mantissaBits) {
        if (!(x.elementType() instanceof FloatType ft)) {
            throw new InterpreterException("reduce_precision of " + x.type().mlir());
        }
        FloatFormat f = ft.format();
        int srcMantissa = f.mantissaBits();
        int srcExponent = f.exponentBits();
        int width = f.bitWidth();
        long[] out = new long[x.size()];
        for (int i = 0; i < out.length; i++) {
            long bits = x.rawBits(i);
            long v = bits;
            if (mantissaBits == 0 && Double.isNaN(f.decode(bits))) {
                // no mantissa left to hold a NaN: it becomes an infinity of the same sign
                out[i] = (bits & (1L << (width - 1))) | (((1L << srcExponent) - 1) << srcMantissa);
                continue;
            }
            if (mantissaBits < srcMantissa) {
                int shift = srcMantissa - mantissaBits;
                long lastBitMask = 1L << shift;
                long baseBias = (lastBitMask >>> 1) - 1;
                long lastBit = (v & lastBitMask) >>> shift;
                v = (v + lastBit + baseBias) & ~(lastBitMask - 1);
            }
            if (exponentBits < srcExponent) {
                long signMask = 1L << (width - 1);
                long expMask = ((1L << srcExponent) - 1) << srcMantissa;
                long bias = (1L << (srcExponent - 1)) - 1;
                long reducedBias = (1L << (exponentBits - 1)) - 1;
                long maxExp = bias + reducedBias;
                long minExp = bias - reducedBias;
                long exp = v & expMask;
                long signedZero = v & signMask;
                if (Long.compareUnsigned(exp, maxExp << srcMantissa) > 0) {
                    v = signedZero | expMask;
                } else if (Long.compareUnsigned(exp, minExp << srcMantissa) <= 0) {
                    v = signedZero;
                }
            }
            long mask = width == 64 ? -1L : (1L << width) - 1;
            v &= mask;
            if (mantissaBits > 0 && Double.isNaN(f.decode(bits))) {
                v = bits;
            }
            out[i] = v;
        }
        return Tensor.ofBits(x.type(), out);
    }
}
