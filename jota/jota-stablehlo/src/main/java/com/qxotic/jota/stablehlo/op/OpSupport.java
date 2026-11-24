package com.qxotic.jota.stablehlo.op;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.attr.UnitAttr;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.List;
import java.util.Optional;
import java.util.OptionalLong;

/** Attribute and operand-segment helpers used by the typed op records. */
public final class OpSupport {

    private OpSupport() {}

    static void checkName(Operation operation, String expected) {
        if (!operation.name().equals(expected)) {
            throw new IllegalArgumentException(
                    "expected " + expected + " operation, got " + operation.name());
        }
    }

    static Attribute required(Operation op, String name) {
        return op.attribute(name)
                .orElseThrow(
                        () ->
                                new IllegalStateException(
                                        op.name() + " is missing attribute '" + name + "'"));
    }

    public static long toLong(Attribute a) {
        return switch (a) {
            case IntegerAttr i -> i.value();
            case BoolAttr b -> b.value() ? 1 : 0;
            default -> throw new IllegalStateException("expected an integer attribute: " + a);
        };
    }

    /**
     * Integer list from {@code array<i64: ...>}, {@code dense<...> : tensor<Nxi64>} or {@code
     * [..]}.
     */
    public static long[] toLongs(Attribute a) {
        return switch (a) {
            case DenseArrayAttr d -> d.values();
            case DenseElementsAttr d -> {
                long[] out = new long[Math.toIntExact(d.numElements())];
                for (int i = 0; i < out.length; i++) {
                    out[i] = d.getLong(i);
                }
                yield out;
            }
            case ArrayAttr arr -> arr.elements().stream().mapToLong(OpSupport::toLong).toArray();
            default -> throw new IllegalStateException("expected an integer list attribute: " + a);
        };
    }

    public static boolean toBool(Attribute a) {
        return switch (a) {
            case BoolAttr b -> b.value();
            case UnitAttr u -> true;
            case IntegerAttr i -> i.value() != 0;
            default -> throw new IllegalStateException("expected a boolean attribute: " + a);
        };
    }

    public static String toStringValue(Attribute a) {
        return switch (a) {
            case StringAttr s -> s.value();
            case SymbolRefAttr s -> s.leaf();
            default -> throw new IllegalStateException("expected a string attribute: " + a);
        };
    }

    static long i64(Operation op, String name) {
        return toLong(required(op, name));
    }

    static OptionalLong i64Opt(Operation op, String name) {
        return op.attribute(name).map(a -> OptionalLong.of(toLong(a))).orElse(OptionalLong.empty());
    }

    static long[] i64s(Operation op, String name) {
        return toLongs(required(op, name));
    }

    static Optional<long[]> i64sOpt(Operation op, String name) {
        return op.attribute(name).map(OpSupport::toLongs);
    }

    static boolean bool(Operation op, String name) {
        return op.attribute(name).map(OpSupport::toBool).orElse(false);
    }

    static String str(Operation op, String name) {
        return toStringValue(required(op, name));
    }

    static Optional<String> strOpt(Operation op, String name) {
        return op.attribute(name).map(OpSupport::toStringValue);
    }

    static double f64(Operation op, String name) {
        Attribute a = required(op, name);
        if (a instanceof FloatAttr f) {
            return f.value();
        }
        throw new IllegalStateException("expected a float attribute: " + a);
    }

    static DenseElementsAttr dense(Operation op, String name) {
        Attribute a = required(op, name);
        if (a instanceof DenseElementsAttr d) {
            return d;
        }
        throw new IllegalStateException("expected a dense elements attribute: " + a);
    }

    static Optional<DenseElementsAttr> denseOpt(Operation op, String name) {
        return op.attribute(name).map(a -> (DenseElementsAttr) a);
    }

    static Type type(Operation op, String name) {
        Attribute a = required(op, name);
        if (a instanceof TypeAttr t) {
            return t.type();
        }
        throw new IllegalStateException("expected a type attribute: " + a);
    }

    /** Operands {@code [from, size - fromEnd)}. */
    static List<Value> slice(Operation op, int from, int fromEnd) {
        return op.operands().subList(from, op.numOperands() - fromEnd);
    }

    /** First or second half of the operands (ops with two equally sized variadic groups). */
    static List<Value> half(Operation op, int which) {
        int n = op.numOperands() / 2;
        return which == 0 ? op.operands().subList(0, n) : op.operands().subList(n, 2 * n);
    }

    /** {@code scatter}: inputs (N), scatter_indices (1), updates (N). */
    static List<Value> scatterGroup(Operation op, int group) {
        int n = (op.numOperands() - 1) / 2;
        return switch (group) {
            case 0 -> op.operands().subList(0, n);
            case 1 -> op.operands().subList(n, n + 1);
            default -> op.operands().subList(n + 1, 2 * n + 1);
        };
    }
}
