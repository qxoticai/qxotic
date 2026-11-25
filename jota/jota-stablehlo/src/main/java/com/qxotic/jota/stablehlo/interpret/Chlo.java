package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

/**
 * The CHLO operations JAX emits ({@code stablehlo/dialect/ChloOps.td}): element-wise special
 * functions computed in double (to about 1e-14 relative error) and rounded to the element type,
 * {@code next_after} on the element format's bits, and {@code top_k}.
 */
final class Chlo {

    private Chlo() {}

    static List<Object> execute(Operation op, List<Object> in) {
        String m = op.mnemonic();
        if (m.equals("top_k")) {
            return topK(op, Interpreter.t(in, 0));
        }
        if (m.equals("next_after")) {
            return List.of(nextAfter(Interpreter.t(in, 0), Interpreter.t(in, 1)));
        }
        DoubleUnaryOperator f =
                switch (m) {
                    case "asin" -> Math::asin;
                    case "acos" -> Math::acos;
                    case "atan" -> Math::atan;
                    case "sinh" -> Math::sinh;
                    case "cosh" -> Math::cosh;
                    case "tan" -> Math::tan;
                    case "asinh" -> Chlo::asinh;
                    case "acosh" -> Chlo::acosh;
                    case "atanh" -> Chlo::atanh;
                    case "erf" -> Chlo::erf;
                    case "erfc" -> Chlo::erfc;
                    case "erf_inv" -> Chlo::erfInv;
                    case "lgamma" -> Chlo::lgamma;
                    case "digamma" -> Chlo::digamma;
                    default -> throw new InterpreterException("chlo." + m + " is not supported");
                };
        Tensor x = Interpreter.t(in, 0);
        if (!x.isFloat()) {
            throw new InterpreterException("chlo." + m + " of " + x.type().mlir());
        }
        double[] r = new double[x.size()];
        for (int i = 0; i < r.length; i++) {
            r[i] = f.applyAsDouble(x.getDouble(i));
        }
        return List.of(Tensor.ofDoubles(Interpreter.resultType(op, 0), r));
    }

    // ================================================================== top_k

    /**
     * The {@code k} largest elements along the last dimension, largest first in IEEE total order
     * (+NaN above +Inf, -NaN below -Inf), equal values in index order, and their indices.
     */
    private static List<Object> topK(Operation op, Tensor x) {
        long k = ((IntegerAttr) Interpreter.attr(op, "k")).value();
        TensorType valuesType = Interpreter.resultType(op, 0);
        TensorType indicesType = Interpreter.resultType(op, 1);
        long[] xd = x.dims();
        int last = xd.length - 1;
        long len = xd[last];
        int rows = (int) (x.size() / Math.max(1, len));
        int[] picked = new int[(int) (rows * k)];
        long[] indices = new long[picked.length];
        for (int row = 0; row < rows; row++) {
            Integer[] order = new Integer[(int) len];
            for (int j = 0; j < len; j++) {
                order[j] = (int) (row * len + j);
            }
            Arrays.sort(order, (a, b) -> -compare(x, a, b)); // stable: ties keep index order
            for (int j = 0; j < k; j++) {
                picked[(int) (row * k + j)] = order[j];
                indices[(int) (row * k + j)] = order[j] - row * len;
            }
        }
        List<Object> out = new ArrayList<>();
        out.add(x.select(valuesType, picked));
        out.add(Tensor.ofLongs(indicesType, indices));
        return out;
    }

    /** IEEE total order for floats (-NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN). */
    private static int compare(Tensor x, int a, int b) {
        if (x.isFloat()) {
            return Long.compare(totalOrderKey(x, a), totalOrderKey(x, b));
        }
        return x.isUnsigned()
                ? Long.compareUnsigned(x.getLong(a), x.getLong(b))
                : Long.compare(x.getLong(a), x.getLong(b));
    }

    private static long totalOrderKey(Tensor x, int i) {
        int w = x.bitWidth();
        long bits = x.rawBits(i);
        long signBit = 1L << (w - 1);
        long magnitude = bits & (signBit - 1);
        return (bits & signBit) != 0 ? -magnitude - 1 : magnitude;
    }

    // ================================================================== next_after

    /** The next value of the element format after {@code x} in the direction of {@code y}. */
    private static Tensor nextAfter(Tensor x, Tensor y) {
        FloatFormat f = ((FloatType) x.elementType()).format();
        int w = f.bitWidth();
        long signBit = 1L << (w - 1);
        long[] out = new long[x.size()];
        for (int i = 0; i < out.length; i++) {
            double a = x.getDouble(i);
            double b = y.getDouble(i);
            long bits = x.rawBits(i);
            if (Double.isNaN(a) || Double.isNaN(b)) {
                out[i] = f.encode(Double.NaN);
            } else if (a == b) {
                out[i] = y.rawBits(i);
            } else if (a == 0) {
                out[i] = (b < 0 ? signBit : 0) | 1; // smallest subnormal toward y
            } else {
                boolean awayFromZero = (a < b) == (a > 0);
                out[i] = awayFromZero ? bits + 1 : bits - 1;
            }
        }
        return Tensor.ofBits(x.type(), out);
    }

    // ================================================================== functions

    private static double asinh(double x) {
        double a = Math.abs(x);
        double r =
                a > 1e8
                        ? Math.log(a) + Math.log(2)
                        : Math.log1p(a + a * a / (1 + Math.sqrt(1 + a * a)));
        return Math.copySign(r, x);
    }

    private static double acosh(double x) {
        if (x < 1) {
            return Double.NaN;
        }
        return x > 1e8
                ? Math.log(x) + Math.log(2)
                : Math.log1p((x - 1) + Math.sqrt((x - 1) * (x + 1)));
    }

    private static double atanh(double x) {
        return 0.5 * Math.log1p(2 * x / (1 - x));
    }

    /** erf by its Taylor series below 2.5, else 1 - erfc. */
    static double erf(double x) {
        if (Double.isNaN(x)) {
            return x;
        }
        double a = Math.abs(x);
        if (a >= 2.5) {
            return Math.copySign(1 - erfcLarge(a), x);
        }
        double sum = 0;
        double term = a;
        for (int n = 0; n < 200; n++) {
            double t = term / (2 * n + 1);
            sum += t;
            if (Math.abs(t) < 1e-17 * Math.abs(sum)) {
                break;
            }
            term *= -a * a / (n + 1);
        }
        return Math.copySign(2 / Math.sqrt(Math.PI) * sum, x);
    }

    static double erfc(double x) {
        if (Double.isNaN(x)) {
            return x;
        }
        if (x >= 2.5) {
            return erfcLarge(x);
        }
        if (x <= -2.5) {
            return 2 - erfcLarge(-x);
        }
        return 1 - erf(x);
    }

    /** erfc for x >= 2.5 by its continued fraction (modified Lentz). */
    private static double erfcLarge(double x) {
        if (Double.isInfinite(x)) {
            return 0;
        }
        // erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
        double tiny = 1e-300;
        double f = x;
        double c = x;
        double d = 0;
        for (int n = 1; n < 500; n++) {
            double an = n / 2.0;
            d = x + an * d;
            d = Math.abs(d) < tiny ? tiny : d;
            c = x + an / c;
            c = Math.abs(c) < tiny ? tiny : c;
            d = 1 / d;
            double delta = c * d;
            f *= delta;
            if (Math.abs(delta - 1) < 1e-16) {
                break;
            }
        }
        return Math.exp(-x * x) / Math.sqrt(Math.PI) / f;
    }

    /** Inverse of erf: an initial guess refined by Newton steps on erf. */
    private static double erfInv(double y) {
        if (Double.isNaN(y) || y < -1 || y > 1) {
            return Double.NaN;
        }
        if (y == 1 || y == -1) {
            return Math.copySign(Double.POSITIVE_INFINITY, y);
        }
        if (y == 0) {
            return y;
        }
        // Giles' single-precision approximation as the starting point
        double w = -Math.log((1 - y) * (1 + y));
        double x;
        if (w < 5) {
            w -= 2.5;
            double p = 2.81022636e-08;
            p = 3.43273939e-07 + p * w;
            p = -3.5233877e-06 + p * w;
            p = -4.39150654e-06 + p * w;
            p = 0.00021858087 + p * w;
            p = -0.00125372503 + p * w;
            p = -0.00417768164 + p * w;
            p = 0.246640727 + p * w;
            p = 1.50140941 + p * w;
            x = p * y;
        } else {
            w = Math.sqrt(w) - 3;
            double p = -0.000200214257;
            p = 0.000100950558 + p * w;
            p = 0.00134934322 + p * w;
            p = -0.00367342844 + p * w;
            p = 0.00573950773 + p * w;
            p = -0.0076224613 + p * w;
            p = 0.00943887047 + p * w;
            p = 1.00167406 + p * w;
            p = 2.83297682 + p * w;
            x = p * y;
        }
        for (int k = 0; k < 4; k++) {
            double err = (Math.abs(y) > 0.9 ? -(erfc(x) - (1 - y)) : erf(x) - y);
            x -= err / (2 / Math.sqrt(Math.PI) * Math.exp(-x * x));
        }
        return x;
    }

    private static final double[] LANCZOS = {
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7
    };

    /** log|Gamma(x)| (Lanczos, g = 7; reflection below 0.5). */
    static double lgamma(double x) {
        if (Double.isNaN(x)) {
            return x;
        }
        if (Double.isInfinite(x)) {
            return Double.POSITIVE_INFINITY;
        }
        if (x <= 0 && x == Math.rint(x)) {
            return Double.POSITIVE_INFINITY; // poles
        }
        if (x < 0.5) {
            return Math.log(Math.PI / Math.abs(Math.sin(Math.PI * x))) - lgamma(1 - x);
        }
        if (x == 1 || x == 2) {
            return 0;
        }
        double z = x - 1;
        double a = LANCZOS[0];
        double t = z + 7.5;
        for (int i = 1; i < 9; i++) {
            a += LANCZOS[i] / (z + i);
        }
        return 0.5 * Math.log(2 * Math.PI) + (z + 0.5) * Math.log(t) - t + Math.log(a);
    }

    /** digamma: recurrence up to 10, then the asymptotic series; reflection below 0. */
    static double digamma(double x) {
        if (Double.isNaN(x) || x == Double.POSITIVE_INFINITY) {
            return x;
        }
        if (x == Double.NEGATIVE_INFINITY) {
            return Double.NaN;
        }
        if (x <= 0 && x == Math.rint(x)) {
            return Double.NaN; // poles (XLA gives NaN)
        }
        if (x < 0) {
            return digamma(1 - x) - Math.PI / Math.tan(Math.PI * x);
        }
        double result = 0;
        while (x < 10) {
            result -= 1 / x;
            x += 1;
        }
        double inv = 1 / (x * x);
        double series =
                inv
                        * (1.0 / 12
                                - inv
                                        * (1.0 / 120
                                                - inv
                                                        * (1.0 / 252
                                                                - inv
                                                                        * (1.0 / 240
                                                                                - inv
                                                                                        * (1.0
                                                                                                / 132)))));
        return result + Math.log(x) - 0.5 / x - series;
    }
}
