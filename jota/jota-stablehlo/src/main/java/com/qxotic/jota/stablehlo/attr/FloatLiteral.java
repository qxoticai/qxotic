package com.qxotic.jota.stablehlo.attr;

import com.qxotic.jota.stablehlo.type.FloatFormat;
import java.math.BigInteger;

/**
 * Float literals spelled as MLIR prints them ({@code printFloatValue} in {@code AsmPrinter.cpp}):
 * six-digit exponential notation ({@code 1.000000e+00}) when it parses back to the same value,
 * otherwise the natural-precision decimal of {@code APFloat::toString} ({@code 0.333333343}),
 * otherwise (infinities, NaNs) the hexadecimal bit pattern ({@code 0x7FC00000}).
 *
 * <p>The decimal conversion is a port of {@code toStringImpl} in LLVM's {@code APFloat.cpp},
 * including its round-half-up truncation, so that the digits match MLIR's exactly.
 */
final class FloatLiteral {

    private FloatLiteral() {}

    static String of(long bits, FloatFormat format) {
        double v = format.decode(bits);
        if (Double.isFinite(v)) {
            String s = toString(v, format, 6, 0, false);
            if (roundTrips(s, bits, format)) {
                return s;
            }
            s = toString(v, format, 0, 3, true);
            if (s.indexOf('.') >= 0) {
                return s;
            }
        }
        return FloatAttr.hex(bits, format.bitWidth());
    }

    private static boolean roundTrips(String s, long bits, FloatFormat format) {
        try {
            return format.encode(Double.parseDouble(s)) == bits;
        } catch (RuntimeException e) {
            return false;
        }
    }

    /** {@code APFloat::toString(Str, FormatPrecision, FormatMaxPadding, TruncateZero)}. */
    static String toString(
            double v,
            FloatFormat format,
            int formatPrecision,
            int formatMaxPadding,
            boolean truncateZero) {
        StringBuilder str = new StringBuilder();
        boolean negative = Double.doubleToRawLongBits(v) < 0;
        if (v == 0) {
            if (negative) {
                str.append('-');
            }
            if (formatMaxPadding == 0) {
                if (truncateZero) {
                    str.append("0.0E+0");
                } else {
                    str.append("0.0");
                    if (formatPrecision > 1) {
                        str.append("0".repeat(formatPrecision - 1));
                    }
                    str.append("e+00");
                }
            } else {
                str.append('0');
            }
            return str.toString();
        }
        if (negative) {
            str.append('-');
        }
        if (formatPrecision == 0) {
            int semanticsPrecision = format.mantissaBits() + 1;
            formatPrecision = 2 + semanticsPrecision * 59 / 196;
        }

        // v = significand * 2^exp with an odd significand (trailing binary zeros dropped)
        long raw = Double.doubleToRawLongBits(Math.abs(v));
        int biased = (int) (raw >>> 52);
        long fraction = raw & ((1L << 52) - 1);
        long mantissa = biased == 0 ? fraction : fraction | (1L << 52);
        int exp = (biased == 0 ? 1 : biased) - 1075;
        int tz = Long.numberOfTrailingZeros(mantissa);
        BigInteger significand = BigInteger.valueOf(mantissa >>> tz);
        exp += tz;

        // change the exponent from 2^e to 10^e
        if (exp > 0) {
            significand = significand.shiftLeft(exp);
            exp = 0;
        } else if (exp < 0) {
            significand = significand.multiply(BigInteger.valueOf(5).pow(-exp));
        }

        // AdjustToPrecision(APInt): drop decimal digits beyond what the precision needs
        int bitsRequired = (formatPrecision * 196 + 58) / 59;
        int bitLength = significand.bitLength();
        if (bitLength > bitsRequired) {
            int tensRemovable = (bitLength - bitsRequired) * 59 / 196;
            if (tensRemovable > 0) {
                exp += tensRemovable;
                significand = significand.divide(BigInteger.TEN.pow(tensRemovable));
            }
        }

        // digits, most significant first, trailing zeros dropped
        String digits = significand.toString();
        int end = digits.length();
        while (end > 1 && digits.charAt(end - 1) == '0') {
            end--;
            exp++;
        }
        StringBuilder buffer = new StringBuilder(digits.substring(0, end));

        // AdjustToPrecision(buffer): round half up to formatPrecision digits
        int n = buffer.length();
        if (n > formatPrecision) {
            int removed = n - formatPrecision;
            if (buffer.charAt(formatPrecision) < '5') {
                buffer.setLength(formatPrecision);
                while (buffer.length() > 1 && buffer.charAt(buffer.length() - 1) == '0') {
                    buffer.setLength(buffer.length() - 1);
                    removed++;
                }
                exp += removed;
            } else {
                buffer.setLength(formatPrecision);
                while (buffer.length() > 0 && buffer.charAt(buffer.length() - 1) == '9') {
                    buffer.setLength(buffer.length() - 1);
                    removed++;
                }
                if (buffer.isEmpty()) {
                    buffer.append('1');
                    exp += n;
                } else {
                    int last = buffer.length() - 1;
                    buffer.setCharAt(last, (char) (buffer.charAt(last) + 1));
                    exp += removed;
                }
            }
        }

        int nDigits = buffer.length();
        boolean scientific;
        if (formatMaxPadding == 0) {
            scientific = true;
        } else if (exp >= 0) {
            scientific = exp > formatMaxPadding || nDigits + exp > formatPrecision;
        } else {
            int msd = exp + nDigits - 1;
            scientific = msd < 0 && -msd > formatMaxPadding;
        }

        if (scientific) {
            exp += nDigits - 1;
            str.append(buffer.charAt(0)).append('.');
            if (nDigits == 1 && truncateZero) {
                str.append('0');
            } else {
                str.append(buffer, 1, nDigits);
            }
            if (!truncateZero && formatPrecision > nDigits - 1) {
                str.append("0".repeat(formatPrecision - nDigits + 1));
            }
            str.append(truncateZero ? 'E' : 'e');
            str.append(exp >= 0 ? '+' : '-');
            String e = Integer.toString(Math.abs(exp));
            if (!truncateZero && e.length() < 2) {
                str.append('0');
            }
            return str.append(e).toString();
        }
        if (exp >= 0) {
            str.append(buffer);
            str.append("0".repeat(exp));
            return str.toString();
        }
        int wholeDigits = exp + nDigits;
        if (wholeDigits > 0) {
            str.append(buffer, 0, wholeDigits).append('.').append(buffer, wholeDigits, nDigits);
        } else {
            str.append("0.").append("0".repeat(-wholeDigits)).append(buffer);
        }
        return str.toString();
    }
}
