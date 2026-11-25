package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;

/**
 * Element-wise semantics of the StableHLO ops ({@code docs/spec.md}). Where the spec leaves a
 * result implementation-defined, XLA's is used, since that is what {@code stablehlo/testdata}
 * expects: integer division by zero gives -1 (all bits set) and remainder by zero the dividend,
 * {@code MIN / -1} gives {@code MIN}, shift amounts of at least the bit width give 0 (or the sign
 * for arithmetic shifts), float to integer conversion saturates (NaN gives 0).
 *
 * <p>Floats are computed in double and rounded once to the element format, which is exact for the
 * basic operations of f32 and narrower formats.
 */
final class Elementwise {

    private Elementwise() {}

    // ================================================================== unary

    static Tensor unary(String op, Tensor x, TensorType resultType) {
        int n = x.size();
        if (x.isComplex()) {
            return complexUnary(op, x, resultType);
        }
        if (op.equals("is_finite")) {
            long[] r = new long[n];
            for (int i = 0; i < n; i++) {
                r[i] = Double.isFinite(x.getDouble(i)) ? 1 : 0;
            }
            return Tensor.ofLongs(resultType, r);
        }
        if (x.isFloat()) {
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                r[i] = floatUnary(op, x.getDouble(i));
            }
            return Tensor.ofDoubles(resultType, r);
        }
        IntegerType it = (IntegerType) x.elementType();
        long[] r = new long[n];
        for (int i = 0; i < n; i++) {
            r[i] = intUnary(op, x.getLong(i), it);
        }
        return Tensor.ofLongs(resultType, r);
    }

    private static double floatUnary(String op, double v) {
        return switch (op) {
            case "abs" -> Math.abs(v);
            case "negate" -> -v;
            case "exponential" -> Math.exp(v);
            case "exponential_minus_one" -> Math.expm1(v);
            case "log" -> Math.log(v);
            case "log_plus_one" -> Math.log1p(v);
            case "logistic" -> 1 / (1 + Math.exp(-v));
            case "sqrt" -> Math.sqrt(v);
            case "rsqrt" -> 1 / Math.sqrt(v);
            case "cbrt" -> Math.cbrt(v);
            case "sine" -> Math.sin(v);
            case "cosine" -> Math.cos(v);
            case "tan" -> Math.tan(v);
            case "tanh" -> Math.tanh(v);
            case "floor" -> Math.floor(v);
            case "ceil" -> Math.ceil(v);
            case "round_nearest_even" -> Math.rint(v);
            case "round_nearest_afz" -> roundHalfAwayFromZero(v);
            case "sign" -> Double.isNaN(v) || v == 0 ? v : Math.signum(v);
            default -> throw new InterpreterException("stablehlo." + op + " on floats");
        };
    }

    private static double roundHalfAwayFromZero(double v) {
        if (!Double.isFinite(v)) {
            return v;
        }
        double t = v < 0 ? Math.ceil(v) : Math.floor(v);
        if (Math.abs(v - t) >= 0.5) {
            t += Math.signum(v);
        }
        return t == 0 ? Math.copySign(0, v) : t;
    }

    private static long intUnary(String op, long v, IntegerType it) {
        int w = it.bitWidth();
        long mask = w == 64 ? -1L : (1L << w) - 1;
        return switch (op) {
            case "abs" -> Math.abs(v);
            case "negate" -> -v;
            case "not" -> it.isBoolean() ? (v == 0 ? 1 : 0) : ~v;
            case "popcnt" -> Long.bitCount(v & mask);
            case "count_leading_zeros" -> Long.numberOfLeadingZeros(v & mask) - (64 - w);
            case "sign" -> it.isUnsigned() ? (v == 0 ? 0 : 1) : Long.signum(v);
            default -> throw new InterpreterException("stablehlo." + op + " on integers");
        };
    }

    private static Tensor complexUnary(String op, Tensor x, TensorType resultType) {
        int n = x.size();
        double[] re = new double[n];
        double[] im = new double[n];
        boolean realResult = !(resultType.elementType() instanceof ComplexType);
        for (int i = 0; i < n; i++) {
            double a = x.getDouble(i);
            double b = x.getImag(i);
            switch (op) {
                case "negate" -> {
                    re[i] = -a;
                    im[i] = -b;
                }
                case "abs" -> re[i] = Math.hypot(a, b);
                case "real" -> re[i] = a;
                case "imag" -> re[i] = b;
                case "exponential" -> {
                    java.util.function.DoubleUnaryOperator r = rounding(x);
                    double e = r.applyAsDouble(Math.exp(a));
                    re[i] = e * r.applyAsDouble(Math.cos(b));
                    im[i] = e * r.applyAsDouble(Math.sin(b));
                }
                case "log" -> {
                    re[i] = Math.log(Math.hypot(a, b));
                    im[i] = Math.atan2(b, a);
                }
                case "sqrt" -> {
                    double m = Math.hypot(a, b);
                    re[i] = Math.sqrt((m + a) / 2);
                    im[i] = Math.copySign(Math.sqrt((m - a) / 2), b);
                }
                case "sign" -> {
                    double m = Math.hypot(a, b);
                    re[i] = m == 0 ? 0 : a / m;
                    im[i] = m == 0 ? 0 : b / m;
                }
                default -> throw new InterpreterException("stablehlo." + op + " on complex");
            }
        }
        return realResult ? Tensor.ofDoubles(resultType, re) : Tensor.ofComplex(resultType, re, im);
    }

    // ================================================================== binary

    static Tensor binary(String op, Tensor x, Tensor y, TensorType resultType) {
        return binary(op, x, y, resultType, Interpreter.Semantics.XLA);
    }

    static Tensor binary(
            String op, Tensor x, Tensor y, TensorType resultType, Interpreter.Semantics sem) {
        int n = x.size();
        if (x.isComplex()) {
            return complexBinary(op, x, y, resultType);
        }
        if (op.equals("complex")) {
            double[] re = new double[n];
            double[] im = new double[n];
            for (int i = 0; i < n; i++) {
                re[i] = x.getDouble(i);
                im[i] = y.getDouble(i);
            }
            return Tensor.ofComplex(resultType, re, im);
        }
        if (x.isFloat()) {
            double[] r = new double[n];
            for (int i = 0; i < n; i++) {
                r[i] = floatBinary(op, x.getDouble(i), y.getDouble(i));
            }
            return Tensor.ofDoubles(resultType, r);
        }
        IntegerType it = (IntegerType) x.elementType();
        long[] r = new long[n];
        for (int i = 0; i < n; i++) {
            r[i] = intBinary(op, x.getLong(i), y.getLong(i), it, sem);
        }
        return Tensor.ofLongs(resultType, r);
    }

    private static double floatBinary(String op, double a, double b) {
        return switch (op) {
            case "add" -> a + b;
            case "subtract" -> a - b;
            case "multiply" -> a * b;
            case "divide" -> a / b;
            case "remainder" -> a % b;
            case "power" -> Math.pow(a, b);
            case "atan2" -> Math.atan2(a, b);
            case "maximum" -> Double.isNaN(a) || Double.isNaN(b) ? Double.NaN : Math.max(a, b);
            case "minimum" -> Double.isNaN(a) || Double.isNaN(b) ? Double.NaN : Math.min(a, b);
            default -> throw new InterpreterException("stablehlo." + op + " on floats");
        };
    }

    private static long intBinary(
            String op, long a, long b, IntegerType it, Interpreter.Semantics sem) {
        boolean reference = sem == Interpreter.Semantics.REFERENCE;
        int w = it.bitWidth();
        boolean unsigned = it.isUnsigned() || it.isBoolean();
        long min = unsigned ? 0 : (w == 64 ? Long.MIN_VALUE : -(1L << (w - 1)));
        return switch (op) {
            case "add" -> it.isBoolean() ? a | b : a + b;
            case "subtract" -> a - b;
            case "multiply" -> it.isBoolean() ? a & b : a * b;
            case "divide" -> {
                if (b == 0) {
                    yield reference ? 0 : -1; // XLA: all bits set
                }
                if (!unsigned && a == min && b == -1) {
                    yield reference ? 0 : min;
                }
                yield unsigned ? Long.divideUnsigned(a, b) : a / b;
            }
            case "remainder" -> {
                if (b == 0) {
                    yield reference ? 0 : a;
                }
                if (!unsigned && a == min && b == -1) {
                    yield 0;
                }
                yield unsigned ? Long.remainderUnsigned(a, b) : a % b;
            }
            case "maximum" ->
                    (unsigned ? Long.compareUnsigned(a, b) : Long.compare(a, b)) >= 0 ? a : b;
            case "minimum" ->
                    (unsigned ? Long.compareUnsigned(a, b) : Long.compare(a, b)) <= 0 ? a : b;
            case "and" -> a & b;
            case "or" -> a | b;
            case "xor" -> a ^ b;
            case "power" -> intPower(a, b, unsigned);
            case "shift_left" -> Long.compareUnsigned(b & mask(w), w) >= 0 ? 0 : a << b;
            case "shift_right_logical" ->
                    Long.compareUnsigned(b & mask(w), w) >= 0 ? 0 : (a & mask(w)) >>> b;
            case "shift_right_arithmetic" ->
                    Long.compareUnsigned(b & mask(w), w) >= 0 ? (a < 0 ? -1 : 0) : a >> b;
            default -> throw new InterpreterException("stablehlo." + op + " on integers");
        };
    }

    private static long mask(int w) {
        return w == 64 ? -1L : (1L << w) - 1;
    }

    /** Exponentiation by squaring; a negative exponent gives 0 except for bases 1 and -1. */
    private static long intPower(long base, long exp, boolean unsigned) {
        if (!unsigned && exp < 0) {
            if (base == 1) {
                return 1;
            }
            if (base == -1) {
                return (exp & 1) == 0 ? 1 : -1;
            }
            return 0;
        }
        long result = 1;
        long b = base;
        long e = exp;
        while (e != 0) {
            if ((e & 1) != 0) {
                result *= b;
            }
            b *= b;
            e >>>= 1;
        }
        return result;
    }

    private static Tensor complexBinary(String op, Tensor x, Tensor y, TensorType resultType) {
        int n = x.size();
        double[] re = new double[n];
        double[] im = new double[n];
        for (int i = 0; i < n; i++) {
            double a = x.getDouble(i);
            double b = x.getImag(i);
            double c = y.getDouble(i);
            double d = y.getImag(i);
            switch (op) {
                case "add" -> {
                    re[i] = a + c;
                    im[i] = b + d;
                }
                case "subtract" -> {
                    re[i] = a - c;
                    im[i] = b - d;
                }
                case "multiply" -> {
                    java.util.function.DoubleUnaryOperator r = rounding(x);
                    re[i] = r.applyAsDouble(a * c) - r.applyAsDouble(b * d);
                    im[i] = r.applyAsDouble(a * d) + r.applyAsDouble(b * c);
                }
                case "divide" -> {
                    java.util.function.DoubleUnaryOperator r = rounding(x);
                    double den = r.applyAsDouble(r.applyAsDouble(c * c) + r.applyAsDouble(d * d));
                    re[i] = r.applyAsDouble(r.applyAsDouble(a * c) + r.applyAsDouble(b * d)) / den;
                    im[i] = r.applyAsDouble(r.applyAsDouble(b * c) - r.applyAsDouble(a * d)) / den;
                }
                default -> throw new InterpreterException("stablehlo." + op + " on complex");
            }
        }
        return Tensor.ofComplex(resultType, re, im);
    }

    /**
     * Rounding of the intermediate products of complex arithmetic: XLA computes complex64 in f32
     * (each product rounded), which the results of testdata reflect beyond their tolerance.
     */
    static java.util.function.DoubleUnaryOperator rounding(Tensor x) {
        return ((ComplexType) x.elementType()).elementType().format() == FloatFormat.F32
                ? v -> (float) v
                : v -> v;
    }

    // ================================================================== compare

    static Tensor compare(String direction, String compareType, Tensor x, Tensor y, TensorType r) {
        int n = x.size();
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            int c;
            boolean unordered = false;
            if (x.isComplex()) {
                boolean eq = x.getDouble(i) == y.getDouble(i) && x.getImag(i) == y.getImag(i);
                c = eq ? 0 : 1;
                if (!direction.equals("EQ") && !direction.equals("NE")) {
                    throw new InterpreterException("ordered comparison of complex numbers");
                }
            } else if (x.isFloat()) {
                double a = x.getDouble(i);
                double b = y.getDouble(i);
                if ("TOTALORDER".equals(compareType)) {
                    c = Long.compare(totalOrderKey(x, i), totalOrderKey(y, i));
                } else if ("WEAKORDER".equals(compareType)) {
                    // every NaN is one value above +Inf; -0 == +0
                    boolean na = Double.isNaN(a);
                    boolean nb = Double.isNaN(b);
                    c = na || nb ? Boolean.compare(na, nb) : (a == b ? 0 : Double.compare(a, b));
                } else {
                    unordered = Double.isNaN(a) || Double.isNaN(b);
                    c = Double.compare(a, b);
                    if (a == b) {
                        c = 0; // -0 == +0
                    }
                }
            } else {
                long a = x.getLong(i);
                long b = y.getLong(i);
                c =
                        x.isUnsigned() || "UNSIGNED".equals(compareType)
                                ? Long.compareUnsigned(a, b)
                                : Long.compare(a, b);
            }
            boolean v =
                    switch (direction) {
                        case "EQ" -> !unordered && c == 0;
                        case "NE" -> unordered || c != 0;
                        case "LT" -> !unordered && c < 0;
                        case "LE" -> !unordered && c <= 0;
                        case "GT" -> !unordered && c > 0;
                        case "GE" -> !unordered && c >= 0;
                        default -> throw new InterpreterException("comparison " + direction);
                    };
            out[i] = v ? 1 : 0;
        }
        return Tensor.ofLongs(r, out);
    }

    /** IEEE total order: sign-magnitude bits mapped to a two's-complement order. */
    private static long totalOrderKey(Tensor x, int i) {
        int w = x.bitWidth();
        long bits = x.rawBits(i);
        long signBit = 1L << (w - 1);
        long magnitude = bits & (signBit - 1);
        return (bits & signBit) != 0 ? -magnitude - 1 : magnitude;
    }

    // ================================================================== convert

    static Tensor convert(Tensor x, TensorType r) {
        int n = x.size();
        var target = r.elementType();
        if (target instanceof ComplexType) {
            double[] re = new double[n];
            double[] im = new double[n];
            for (int i = 0; i < n; i++) {
                re[i] = x.getDouble(i);
                im[i] = x.isComplex() ? x.getImag(i) : 0;
            }
            return Tensor.ofComplex(r, re, im);
        }
        if (target instanceof FloatType) {
            double[] out = new double[n];
            FloatFormat f = ((FloatType) target).format();
            long[] bits = new long[n];
            for (int i = 0; i < n; i++) {
                if (x.isInteger() && !x.isUnsigned() && f == FloatFormat.F32) {
                    bits[i] = Float.floatToRawIntBits((float) x.getLong(i)) & 0xFFFFFFFFL;
                    continue; // single rounding from 64-bit integers
                }
                out[i] = x.getDouble(i);
                bits[i] = f.encode(out[i]);
            }
            return Tensor.ofBits(r, bits);
        }
        IntegerType it = (IntegerType) target;
        long[] out = new long[n];
        for (int i = 0; i < n; i++) {
            if (it.isBoolean()) {
                // complex to real types keeps the real part only
                out[i] =
                        x.isFloat() || x.isComplex()
                                ? (x.getDouble(i) != 0 ? 1 : 0)
                                : (x.getLong(i) != 0 ? 1 : 0);
            } else if (x.isFloat() || x.isComplex()) {
                out[i] = saturate(x.getDouble(i), it);
            } else {
                out[i] = x.getLong(i);
            }
        }
        return Tensor.ofLongs(r, out);
    }

    /** Float to integer, truncating toward zero and saturating at the type's range (NaN is 0). */
    private static long saturate(double v, IntegerType it) {
        if (Double.isNaN(v)) {
            return 0;
        }
        int w = it.bitWidth();
        if (it.isUnsigned()) {
            if (v <= 0) {
                return 0;
            }
            if (w == 64) {
                return v >= 18446744073709551615.0 ? -1L : new java.math.BigDecimal(v).longValue();
            }
            double max = (double) ((1L << w) - 1);
            return v >= max ? (1L << w) - 1 : (long) v;
        }
        if (w == 64) {
            return (long) v; // Java saturates
        }
        long max = (1L << (w - 1)) - 1;
        long min = -(1L << (w - 1));
        if (v >= max) {
            return max;
        }
        if (v <= min) {
            return min;
        }
        return (long) v;
    }
}
