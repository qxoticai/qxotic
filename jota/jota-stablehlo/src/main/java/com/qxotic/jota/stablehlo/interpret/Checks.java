package com.qxotic.jota.stablehlo.interpret;

import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import java.util.List;

/**
 * The {@code check} dialect of StableHLO's interpreter tests ({@code stablehlo/tests/CheckOps.td},
 * {@code CheckOps.cpp}) and the {@code check.expect_*} custom calls of {@code testdata} (as {@code
 * stablehlo-translate --interpret} runs them).
 *
 * <ul>
 *   <li>{@code expect_eq}, {@code expect_eq_const}: equal elements (floats bitwise, NaNs equal);
 *   <li>{@code expect_almost_eq}, {@code expect_almost_eq_const}: absolute difference at most
 *       {@code tolerance} (default 1e-4), as {@code areApproximatelyEqual};
 *   <li>{@code expect_close}: at most {@code max_ulp_difference} units in the last place (default
 *       1; the custom call allows 3).
 * </ul>
 */
final class Checks {

    private Checks() {}

    static void check(String name, Operation op, List<Object> in) {
        boolean customCall = op.name().equals("stablehlo.custom_call");
        Tensor actual = (Tensor) in.get(0);
        Tensor expected =
                name.endsWith("_const")
                        ? Tensor.of((DenseElementsAttr) Interpreter.attr(op, "value"))
                        : (Tensor) in.get(1);
        if (actual.size() != expected.size()) {
            throw new Interpreter.CheckFailure(
                    name + ": " + actual.size() + " elements vs " + expected.size());
        }
        switch (name) {
            case "check.expect_eq", "check.expect_eq_const" -> {
                for (int i = 0; i < actual.size(); i++) {
                    if (!sameBits(actual, expected, i)) {
                        fail(name, i, actual, expected);
                    }
                }
            }
            case "check.expect_almost_eq", "check.expect_almost_eq_const" -> {
                double tolerance = doubleAttr(op, "tolerance", 1e-4);
                for (int i = 0; i < actual.size(); i++) {
                    if (!almostEqual(actual, expected, i, tolerance)) {
                        fail(name, i, actual, expected);
                    }
                }
            }
            case "check.expect_close" -> {
                long max = longAttr(op, "max_ulp_difference", customCall ? 3 : 1);
                long min = longAttr(op, "min_ulp_difference", 0);
                for (int i = 0; i < actual.size(); i++) {
                    long d = ulp(actual, expected, i);
                    if (d > max || d < min) {
                        fail(
                                name + " (ULP difference " + Long.toUnsignedString(d) + ")",
                                i,
                                actual,
                                expected);
                    }
                }
            }
            default -> throw new InterpreterException(name + " is not supported");
        }
    }

    private static void fail(String name, int i, Tensor actual, Tensor expected) {
        throw new Interpreter.CheckFailure(
                name
                        + " at element "
                        + i
                        + ": got "
                        + element(actual, i)
                        + ", expected "
                        + element(expected, i));
    }

    private static String element(Tensor t, int i) {
        return t.toDense().elementLiteral(i);
    }

    private static boolean sameBits(Tensor a, Tensor b, int i) {
        if (a.isFloat()) {
            double x = a.getDouble(i);
            double y = b.getDouble(i);
            if (Double.isNaN(x) && Double.isNaN(y)) {
                return true;
            }
        }
        if (a.isComplex() && a.rawImag(i) != b.rawImag(i)) {
            return false;
        }
        return a.isInteger() ? a.getLong(i) == b.getLong(i) : a.rawBits(i) == b.rawBits(i);
    }

    private static boolean almostEqual(Tensor a, Tensor b, int i, double tolerance) {
        if (!a.isFloat() && !a.isComplex()) {
            return a.getLong(i) == b.getLong(i);
        }
        boolean re = approximately(a.getDouble(i), b.getDouble(i), tolerance);
        return a.isComplex() ? re && approximately(a.getImag(i), b.getImag(i), tolerance) : re;
    }

    /** {@code areApproximatelyEqual} of {@code reference/Element.cpp}. */
    private static boolean approximately(double f, double g, double tolerance) {
        if (f == g) {
            return true;
        }
        if (Double.isNaN(f) || Double.isNaN(g)) {
            return Double.isNaN(f) == Double.isNaN(g);
        }
        if (!Double.isFinite(f) || !Double.isFinite(g)) {
            return false;
        }
        return Math.abs(f - g) <= tolerance;
    }

    /** {@code ULPDifference} of {@code CheckOps.cpp} (complex: the larger of both parts). */
    private static long ulp(Tensor a, Tensor b, int i) {
        if (!a.isFloat() && !a.isComplex()) {
            return a.getLong(i) == b.getLong(i) ? 0 : -1;
        }
        int w = a.isComplex() ? a.bitWidth() / 2 : a.bitWidth();
        long re = ulp(a.rawBits(i), b.rawBits(i), a.getDouble(i), b.getDouble(i), w);
        if (!a.isComplex()) {
            return re;
        }
        long im = ulp(a.rawImag(i), b.rawImag(i), a.getImag(i), b.getImag(i), w);
        return Long.compareUnsigned(re, im) >= 0 ? re : im;
    }

    private static long ulp(long fb, long gb, double f, double g, int width) {
        if (fb == gb) {
            return 0;
        }
        if (Double.isFinite(f) && Double.isFinite(g)) {
            long signBit = 1L << (width - 1);
            long af = fb & (signBit - 1);
            long ag = gb & (signBit - 1);
            boolean nf = (fb & signBit) != 0;
            boolean ng = (gb & signBit) != 0;
            if (nf != ng) {
                return af + ag;
            }
            return af > ag ? af - ag : ag - af;
        }
        if (Double.isNaN(f) && Double.isNaN(g)) {
            return 0;
        }
        return -1; // maximal: finite against non-finite
    }

    private static double doubleAttr(Operation op, String name, double fallback) {
        Attribute a = op.attribute(name).orElse(null);
        return switch (a) {
            case FloatAttr f -> f.value();
            case IntegerAttr i -> i.value();
            case null, default -> fallback;
        };
    }

    private static long longAttr(Operation op, String name, long fallback) {
        return op.attribute(name).orElse(null) instanceof IntegerAttr i ? i.value() : fallback;
    }
}
