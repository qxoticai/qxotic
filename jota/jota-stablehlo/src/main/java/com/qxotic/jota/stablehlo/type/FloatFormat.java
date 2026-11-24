package com.qxotic.jota.stablehlo.type;

import java.util.HashMap;
import java.util.Map;

/**
 * The floating-point formats of StableHLO's {@code FloatType}, with exact bit-level encoding and
 * decoding.
 *
 * <p>Every format other than {@code f64} is a subset of {@code double}, so {@link #encode(double)}
 * rounds exactly (to nearest, ties to even) using {@link Math#rint} on power-of-two scaled values.
 *
 * <p>Special-value conventions follow LLVM {@code APFloat} semantics referenced by the spec:
 *
 * <ul>
 *   <li>{@link Kind#IEEE}: infinities and NaNs with an all-ones exponent;
 *   <li>{@link Kind#FN}: finite, NaN is the all-ones pattern (sign bit free);
 *   <li>{@link Kind#FNUZ}: finite, no negative zero, NaN is the sign-only pattern {@code 0x80};
 *   <li>{@link Kind#FNU}: unsigned, finite, NaN is the all-ones pattern;
 *   <li>{@link Kind#FINITE}: no infinity and no NaN (MX fp4/fp6 formats);
 *   <li>{@link Kind#E8M0}: unsigned powers of two, NaN is {@code 0xFF}, no zero.
 * </ul>
 */
public enum FloatFormat {
    F4E2M1FN("f4E2M1FN", 4, 2, 1, 1, Kind.FINITE),
    F6E2M3FN("f6E2M3FN", 6, 2, 3, 1, Kind.FINITE),
    F6E3M2FN("f6E3M2FN", 6, 3, 2, 3, Kind.FINITE),
    F8E3M4("f8E3M4", 8, 3, 4, 3, Kind.IEEE),
    F8E4M3("f8E4M3", 8, 4, 3, 7, Kind.IEEE),
    F8E4M3FN("f8E4M3FN", 8, 4, 3, 7, Kind.FN),
    F8E4M3FNUZ("f8E4M3FNUZ", 8, 4, 3, 8, Kind.FNUZ),
    F8E4M3B11FNUZ("f8E4M3B11FNUZ", 8, 4, 3, 11, Kind.FNUZ),
    F8E5M2("f8E5M2", 8, 5, 2, 15, Kind.IEEE),
    F8E5M2FNUZ("f8E5M2FNUZ", 8, 5, 2, 16, Kind.FNUZ),
    F8E5M3FNU("f8E5M3FNU", 8, 5, 3, 15, Kind.FNU),
    F8E8M0FNU("f8E8M0FNU", 8, 8, 0, 127, Kind.E8M0),
    BF16("bf16", 16, 8, 7, 127, Kind.IEEE),
    F16("f16", 16, 5, 10, 15, Kind.IEEE),
    TF32("tf32", 19, 8, 10, 127, Kind.IEEE),
    F32("f32", 32, 8, 23, 127, Kind.IEEE),
    F64("f64", 64, 11, 52, 1023, Kind.IEEE);

    /** Special-value convention of a format. */
    public enum Kind {
        IEEE,
        FN,
        FNUZ,
        FNU,
        FINITE,
        E8M0
    }

    private static final Map<String, FloatFormat> BY_NAME = new HashMap<>();

    static {
        for (FloatFormat f : values()) {
            BY_NAME.put(f.mlirName, f);
        }
    }

    private final String mlirName;
    private final int bitWidth;
    private final int exponentBits;
    private final int mantissaBits;
    private final int bias;
    private final Kind kind;

    FloatFormat(
            String mlirName,
            int bitWidth,
            int exponentBits,
            int mantissaBits,
            int bias,
            Kind kind) {
        this.mlirName = mlirName;
        this.bitWidth = bitWidth;
        this.exponentBits = exponentBits;
        this.mantissaBits = mantissaBits;
        this.bias = bias;
        this.kind = kind;
    }

    /** Looks up a format by its MLIR spelling ({@code f32}, {@code f8E4M3FN}...), or null. */
    public static FloatFormat byName(String name) {
        return BY_NAME.get(name);
    }

    public String mlirName() {
        return mlirName;
    }

    public int bitWidth() {
        return bitWidth;
    }

    public int exponentBits() {
        return exponentBits;
    }

    public int mantissaBits() {
        return mantissaBits;
    }

    public int bias() {
        return bias;
    }

    public Kind kind() {
        return kind;
    }

    public boolean isSigned() {
        return kind != Kind.FNU && kind != Kind.E8M0;
    }

    public boolean hasInfinity() {
        return kind == Kind.IEEE;
    }

    public boolean hasNaN() {
        return kind != Kind.FINITE;
    }

    /** Storage bytes used by dense element buffers: the bit width rounded up to whole bytes. */
    public int storageBytes() {
        return (bitWidth + 7) / 8;
    }

    private long mask() {
        return bitWidth == 64 ? -1L : (1L << bitWidth) - 1;
    }

    private long signBit() {
        return isSigned() ? 1L << (bitWidth - 1) : 0;
    }

    private long maxExponentField() {
        return (1L << exponentBits) - 1;
    }

    private long mantissaMask() {
        return (1L << mantissaBits) - 1;
    }

    /** The canonical quiet NaN encoding. */
    public long nanBits() {
        return switch (kind) {
            case IEEE ->
                    (maxExponentField() << mantissaBits)
                            | (mantissaBits == 0 ? 0 : 1L << (mantissaBits - 1));
            case FN -> mask() & ~signBit();
            case FNUZ -> signBit();
            case FNU, E8M0 -> mask();
            case FINITE -> throw new IllegalArgumentException(mlirName + " cannot represent NaN");
        };
    }

    /** Decodes raw bits (low {@link #bitWidth()} bits) to a double, exactly. */
    public double decode(long bits) {
        bits &= mask();
        if (this == F64) {
            return Double.longBitsToDouble(bits);
        }
        if (this == F32) {
            return Float.intBitsToFloat((int) bits);
        }
        if (kind == Kind.E8M0) {
            return bits == 0xFF ? Double.NaN : Math.scalb(1.0, (int) bits - bias);
        }
        boolean negative = (bits & signBit()) != 0;
        long exponent = (bits >>> mantissaBits) & maxExponentField();
        long mantissa = bits & mantissaMask();
        switch (kind) {
            case IEEE -> {
                if (exponent == maxExponentField()) {
                    if (mantissa != 0) {
                        return Double.NaN;
                    }
                    return negative ? Double.NEGATIVE_INFINITY : Double.POSITIVE_INFINITY;
                }
            }
            case FN -> {
                if (exponent == maxExponentField() && mantissa == mantissaMask()) {
                    return Double.NaN;
                }
            }
            case FNUZ -> {
                if (bits == signBit()) {
                    return Double.NaN;
                }
            }
            case FNU -> {
                if (bits == mask()) {
                    return Double.NaN;
                }
            }
            default -> {}
        }
        double magnitude;
        if (exponent == 0) {
            magnitude = Math.scalb((double) mantissa, 1 - bias - mantissaBits);
        } else {
            magnitude =
                    Math.scalb(
                            (double) (mantissa | (1L << mantissaBits)),
                            (int) exponent - bias - mantissaBits);
        }
        return negative ? -magnitude : magnitude;
    }

    /**
     * Encodes a double into this format, rounding to nearest with ties to even.
     *
     * <p>Overflow produces infinity for IEEE formats, NaN for formats with NaN but no infinity, and
     * saturates for {@link Kind#FINITE} formats. For unsigned formats negative values become NaN
     * and {@code -0.0} becomes {@code +0.0} (spec of {@code convert}).
     */
    public long encode(double value) {
        if (this == F64) {
            return Double.doubleToRawLongBits(value);
        }
        if (Double.isNaN(value)) {
            return nanBits();
        }
        boolean negative = (Double.doubleToRawLongBits(value) & Long.MIN_VALUE) != 0;
        double magnitude = Math.abs(value);
        if (!isSigned() && negative) {
            if (magnitude == 0.0 && kind != Kind.E8M0) {
                return 0;
            }
            return nanBits();
        }
        if (kind == Kind.E8M0) {
            return encodeE8M0(magnitude);
        }
        long sign = negative ? signBit() : 0;
        if (magnitude == 0.0) {
            return kind == Kind.FNUZ ? 0 : sign;
        }
        if (Double.isInfinite(magnitude)) {
            return overflow(sign);
        }
        int minNormalExponent = 1 - bias;
        int exponent = Math.getExponent(magnitude);
        long field;
        if (exponent < minNormalExponent) {
            // Subnormal range: a result of 2^mantissaBits naturally encodes the smallest normal.
            field = (long) Math.rint(Math.scalb(magnitude, mantissaBits - minNormalExponent));
        } else {
            long q = (long) Math.rint(Math.scalb(magnitude, mantissaBits - exponent));
            if (q == 1L << (mantissaBits + 1)) {
                exponent++;
                q = 1L << mantissaBits;
            }
            long exponentField = (long) exponent + bias;
            field = (exponentField << mantissaBits) | (q & mantissaMask());
            if (isOverflow(exponentField, q & mantissaMask())) {
                return overflow(sign);
            }
        }
        if (field == 0 && kind == Kind.FNUZ) {
            return 0; // underflow to zero: FNUZ has no negative zero (0x80 is NaN)
        }
        return sign | field;
    }

    private boolean isOverflow(long exponentField, long mantissa) {
        long maxField = maxExponentField();
        return switch (kind) {
            case IEEE -> exponentField >= maxField;
            case FN, FNU ->
                    exponentField > maxField
                            || (exponentField == maxField && mantissa == mantissaMask());
            default -> exponentField > maxField;
        };
    }

    private long overflow(long sign) {
        return switch (kind) {
            case IEEE -> sign | (maxExponentField() << mantissaBits);
            case FINITE -> sign | (mask() & ~signBit());
            default -> nanBits();
        };
    }

    private long encodeE8M0(double magnitude) {
        if (magnitude == 0.0 || Double.isInfinite(magnitude)) {
            return nanBits();
        }
        int exponent = Math.getExponent(magnitude);
        double low = Math.scalb(1.0, exponent);
        if (magnitude - low > Math.scalb(1.0, exponent + 1) - magnitude) {
            exponent++;
        } else if (magnitude - low == Math.scalb(1.0, exponent + 1) - magnitude
                && ((exponent + bias) & 1) != 0) {
            exponent++;
        }
        long field = (long) exponent + bias;
        if (field < 0) {
            return 0;
        }
        if (field >= 0xFF) {
            return nanBits();
        }
        return field;
    }

    @Override
    public String toString() {
        return mlirName;
    }
}
