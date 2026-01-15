
package ai.qxotic.jota;

public final class BFloat16 {

    private static final int ROUNDING_BIAS = 0x7fff;

    private BFloat16() {
    }

    public static short fromFloat(float value) {
        int bits = Float.floatToRawIntBits(value);
        int exponent = bits & 0x7f800000;
        int mantissa = bits & 0x007fffff;
        if (exponent == 0x7f800000 && mantissa != 0) {
            int upper = bits >>> 16;
            upper |= 0x0040;
            return (short) upper;
        }
        int lsb = (bits >> 16) & 1;
        int rounded = bits + ROUNDING_BIAS + lsb;
        return (short) (rounded >>> 16);
    }

    public static short fromFloat16(short float16Bits) {
        int bits = float16Bits & 0xffff;
        int sign = bits & 0x8000;
        int exponent = bits & 0x7c00;
        int mantissa = bits & 0x03ff;
        if (exponent == 0x7c00) {
            int bfExponent = 0x7f80;
            int bfMantissa = 0;
            if (mantissa != 0) {
                bfMantissa = mantissa >> 3;
                if (bfMantissa == 0) {
                    bfMantissa = 1;
                }
            }
            return (short) (sign | bfExponent | bfMantissa);
        }
        return fromFloat(Float.float16ToFloat(float16Bits));
    }

    public static float toFloat(short bfloat16) {
        int bits = (bfloat16 & 0xffff) << 16;
        return Float.intBitsToFloat(bits);
    }

    public static short toFloat16(short bfloat16Bits) {
        int bits = bfloat16Bits & 0xffff;
        int sign = bits & 0x8000;
        int exponent = bits & 0x7f80;
        int mantissa = bits & 0x007f;
        if (exponent == 0x7f80) {
            int halfExponent = 0x7c00;
            int halfMantissa = 0;
            if (mantissa != 0) {
                halfMantissa = mantissa << 3;
                if (halfMantissa == 0) {
                    halfMantissa = 1;
                }
            }
            return (short) (sign | halfExponent | halfMantissa);
        }
        return Float.floatToFloat16(toFloat(bfloat16Bits));
    }
}
