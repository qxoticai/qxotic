package com.qxotic.jota.stablehlo.verify;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.ComparisonType;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.FftType;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.op.*;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.FutureType;
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.QuantizedType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Checks the constraints of {@code docs/spec.md} ("Constraints" sections, {@code (C1)}...) for
 * every operation of a program, plus structural rules (function returns and calls, region
 * signatures).
 *
 * <p>Checks are conservative: dimensions compare as compatible when either side is dynamic,
 * signless and signed integers are the same type, tensor encodings (bounds) are left to type
 * inference, and for quantized element types only the storage and expressed types are compared
 * unless the spec states a quantization rule. Constraints that need execution context (number of
 * replicas or partitions) are not checked.
 *
 * <p>An op that passes its own checks is then compared with {@link TypeInference}: its result types
 * must be compatible with the inferred ones (constraint {@code "infer"}), as the {@code
 * InferTypeOpInterface} verification of MLIR does; this is where bounds are checked.
 */
public final class Verifier implements OpVisitor<Void> {

    private final Program program;
    private final List<Diagnostic> diagnostics = new ArrayList<>();
    private Operation current;

    private Verifier(Program program) {
        this.program = program;
    }

    /** All constraint violations of {@code program}, in operation order. */
    public static List<Diagnostic> verify(Program program) {
        Verifier v = new Verifier(program);
        program.operation()
                .walk(
                        op -> {
                            v.current = op;
                            int before = v.diagnostics.size();
                            try {
                                v.common(op);
                                op.asOp().accept(v);
                            } catch (RuntimeException e) {
                                v.fail("attr", "malformed operation: " + e.getMessage());
                            }
                            if (v.diagnostics.size() == before) {
                                v.checkInferredTypes(op);
                            }
                        });
        return List.copyOf(v.diagnostics);
    }

    /**
     * Result types against {@link TypeInference}, for ops that passed their own checks: catches
     * what the constraints leave to inference (bounds, dynamic dimensions, element types).
     */
    private void checkInferredTypes(Operation op) {
        try {
            for (String m : TypeInference.mismatches(op)) {
                fail("infer", m);
            }
        } catch (TypeInference.TypeInferenceException e) {
            fail("infer", e.getMessage());
        }
    }

    @Override
    public Void visitDefault(Op op) {
        return null; // no constraint implemented
    }

    // ================================================================== checks shared by ops

    private static final Set<String> LOGICAL =
            Set.of("stablehlo.and", "stablehlo.or", "stablehlo.xor", "stablehlo.not");
    private static final Set<String> INTEGER_ONLY =
            Set.of(
                    "stablehlo.shift_left",
                    "stablehlo.shift_right_arithmetic",
                    "stablehlo.shift_right_logical",
                    "stablehlo.popcnt",
                    "stablehlo.count_leading_zeros");
    private static final Set<String> FLOAT_COMPLEX =
            Set.of(
                    "stablehlo.exponential",
                    "stablehlo.exponential_minus_one",
                    "stablehlo.log",
                    "stablehlo.log_plus_one",
                    "stablehlo.logistic",
                    "stablehlo.sqrt",
                    "stablehlo.rsqrt",
                    "stablehlo.cbrt",
                    "stablehlo.sine",
                    "stablehlo.cosine",
                    "stablehlo.tan",
                    "stablehlo.tanh");
    private static final Set<String> FLOAT_ONLY =
            Set.of(
                    "stablehlo.ceil",
                    "stablehlo.floor",
                    "stablehlo.round_nearest_afz",
                    "stablehlo.round_nearest_even");

    /** Operand type classes of the op definitions, and the {@code result_accuracy} attribute. */
    private void common(Operation op) {
        String name = op.name();
        for (Value v : op.operands()) {
            TensorType t = tt(v);
            if (t == null) {
                continue;
            }
            ElementType e = t.elementType();
            if (LOGICAL.contains(name)) {
                check(
                        e instanceof IntegerType,
                        "type",
                        "operands must be boolean or integer tensors, but got " + t.mlir());
            } else if (INTEGER_ONLY.contains(name)) {
                check(
                        isInteger(e),
                        "type",
                        "operands must be integer tensors, but got " + t.mlir());
            } else if (FLOAT_COMPLEX.contains(name)) {
                check(
                        e instanceof FloatType
                                || e instanceof ComplexType
                                || e instanceof QuantizedType,
                        "type",
                        "operands must be floating-point, complex or quantized tensors, but got "
                                + t.mlir());
            } else if (FLOAT_ONLY.contains(name)) {
                check(
                        e instanceof FloatType || e instanceof QuantizedType,
                        "type",
                        "operands must be floating-point or quantized tensors, but got "
                                + t.mlir());
            }
        }
        if (op.attribute("result_accuracy").orElse(null)
                        instanceof com.qxotic.jota.stablehlo.attr.DialectAttr ra
                && ra.hasParams()) {
            String mode = ra.param("mode").map(Verifier::enumName).orElse("DEFAULT");
            boolean zero =
                    ra.param("atol").map(Verifier::isZero).orElse(true)
                            && ra.param("rtol").map(Verifier::isZero).orElse(true)
                            && ra.param("ulps").map(Verifier::isZero).orElse(true);
            check(
                    !(mode.equals("DEFAULT") || mode.equals("HIGHEST")) || zero,
                    "accuracy",
                    "tolerances of result_accuracy with mode " + mode + " must be all zero");
        }
    }

    private static String enumName(Attribute a) {
        java.util.regex.Matcher m =
                java.util.regex.Pattern.compile("([A-Z_]+)>*$").matcher(a.toString().trim());
        return m.find() ? m.group(1) : a.toString();
    }

    private static boolean isZero(Attribute a) {
        return switch (a) {
            case com.qxotic.jota.stablehlo.attr.FloatAttr f -> f.value() == 0;
            case com.qxotic.jota.stablehlo.attr.IntegerAttr i -> i.value() == 0;
            default -> true;
        };
    }

    // ================================================================== reporting

    private void fail(String constraint, String message) {
        diagnostics.add(new Diagnostic(current, constraint, message));
    }

    private boolean check(boolean ok, String constraint, String message) {
        if (!ok) {
            fail(constraint, message);
        }
        return ok;
    }

    // ================================================================== type helpers

    static final long DYN = TensorType.DYNAMIC;

    private static TensorType tt(Value v) {
        return v.type() instanceof TensorType t ? t : null;
    }

    private static TensorType tt(Type t) {
        return t instanceof TensorType tensor ? tensor : null;
    }

    static boolean dimEq(long a, long b) {
        return a == b || a == DYN || b == DYN;
    }

    static boolean shapeEq(TensorType a, TensorType b) {
        if (a.rank() != b.rank()) {
            return false;
        }
        for (int i = 0; i < a.rank(); i++) {
            if (!dimEq(a.dim(i), b.dim(i))) {
                return false;
            }
        }
        return true;
    }

    /** Element types equal up to integer signedness spelling and quantization parameters. */
    static boolean elemEq(ElementType a, ElementType b) {
        if (a instanceof IntegerType x && b instanceof IntegerType y) {
            return x.bitWidth() == y.bitWidth() && x.isUnsigned() == y.isUnsigned();
        }
        if (a instanceof QuantizedType x && b instanceof QuantizedType y) {
            return elemEq(x.storageType(), y.storageType())
                    && x.expressedType().equals(y.expressedType());
        }
        return a.equals(b);
    }

    static boolean typeEq(Type a, Type b) {
        if (a instanceof TensorType x && b instanceof TensorType y) {
            return shapeEq(x, y) && elemEq(x.elementType(), y.elementType());
        }
        if (a instanceof TupleType x && b instanceof TupleType y) {
            if (x.types().size() != y.types().size()) {
                return false;
            }
            for (int i = 0; i < x.types().size(); i++) {
                if (!typeEq(x.types().get(i), y.types().get(i))) {
                    return false;
                }
            }
            return true;
        }
        return a.equals(b);
    }

    private static boolean isQuantized(TensorType t) {
        return t != null && t.elementType() instanceof QuantizedType;
    }

    private static boolean isBool(ElementType e) {
        return e instanceof IntegerType it && it.isBoolean();
    }

    private static boolean isInteger(ElementType e) {
        return e instanceof IntegerType it && !it.isBoolean();
    }

    private static int bits(ElementType e) {
        return switch (e) {
            case IntegerType it -> it.bitWidth();
            case FloatType f -> f.format().bitWidth();
            case ComplexType c -> 2 * c.elementType().format().bitWidth();
            case QuantizedType q -> q.storageType().bitWidth();
            case IndexType i -> 64;
        };
    }

    /** {@code is_promotable(x, y)}: same kind, no narrowing. */
    static boolean promotable(ElementType from, ElementType to) {
        if (from instanceof QuantizedType || to instanceof QuantizedType) {
            return true;
        }
        if (isBool(from) || isBool(to)) {
            return isBool(from) && isBool(to);
        }
        if (from instanceof IntegerType a && to instanceof IntegerType b) {
            return a.isUnsigned() == b.isUnsigned() && a.bitWidth() <= b.bitWidth();
        }
        if (from instanceof FloatType a && to instanceof FloatType b) {
            return a.format().bitWidth() <= b.format().bitWidth();
        }
        if (from instanceof ComplexType a && to instanceof ComplexType b) {
            return a.elementType().format().bitWidth() <= b.elementType().format().bitWidth();
        }
        return false;
    }

    private static boolean unique(long[] values) {
        Set<Long> seen = new HashSet<>();
        for (long v : values) {
            if (!seen.add(v)) {
                return false;
            }
        }
        return true;
    }

    private static boolean sorted(long[] values) {
        for (int i = 1; i < values.length; i++) {
            if (values[i] < values[i - 1]) {
                return false;
            }
        }
        return true;
    }

    private static boolean inRange(long[] values, long lo, long hi) {
        for (long v : values) {
            if (v < lo || v >= hi) {
                return false;
            }
        }
        return true;
    }

    private static boolean contains(long[] values, long v) {
        for (long x : values) {
            if (x == v) {
                return true;
            }
        }
        return false;
    }

    private static long[] concat(long[] a, long[] b) {
        long[] out = Arrays.copyOf(a, a.length + b.length);
        System.arraycopy(b, 0, out, a.length, b.length);
        return out;
    }

    private static String str(long[] v) {
        return Arrays.toString(v);
    }

    // ================================================================== region helpers

    private static Block entry(Region r) {
        return r.blocks().size() == 1 ? r.block() : null;
    }

    /** Types returned by the terminator of a single-block region, or null. */
    private static List<Type> returned(Region r) {
        Block b = entry(r);
        if (b == null || b.terminator() == null) {
            return null;
        }
        return b.terminator().operandTypes();
    }

    private static List<Type> argumentTypes(Region r) {
        Block b = entry(r);
        return b == null ? null : b.arguments().stream().map(Value::type).toList();
    }

    private static boolean typesEq(List<Type> a, List<Type> b) {
        if (a == null || b == null || a.size() != b.size()) {
            return false;
        }
        for (int i = 0; i < a.size(); i++) {
            if (!typeEq(a.get(i), b.get(i))) {
                return false;
            }
        }
        return true;
    }

    /**
     * A reduction body, checked as the reference implementation does ({@code verifyReducerShape}):
     * {@code 2N} arguments and {@code N} results; argument {@code i} has the type of result {@code
     * i}, argument {@code N+i} its shape and kind; init value {@code i} has the shape of result
     * {@code i} and promotes to it; input element {@code i} promotes to argument {@code N+i};
     * arguments {@code N+i} have at most the dimensions {@code allowed}, in order. Floating-point
     * precision is ignored in these comparisons, as there. Returns the result element types {@code
     * Ei}, or null after reporting.
     */
    private List<ElementType> checkReducer(
            Region body,
            List<ElementType> inputs,
            List<TensorType> inits,
            long[] allowed,
            String c) {
        int n = inputs.size();
        List<Type> args = argumentTypes(body);
        List<Type> rets = returned(body);
        if (!check(
                        args != null && args.size() == 2 * n,
                        c,
                        "body must take " + 2 * n + " arguments")
                || !check(
                        rets != null && rets.size() == n, c, "body must return " + n + " values")) {
            return null;
        }
        List<ElementType> es = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            TensorType r = tt(rets.get(i));
            TensorType a = tt(args.get(i));
            TensorType b = tt(args.get(n + i));
            if (!check(
                    r != null && a != null && b != null,
                    c,
                    "body arguments and results must be tensors")) {
                return null;
            }
            if (!check(
                            shapeEq(a, r) && elemEq(a.elementType(), r.elementType()),
                            c,
                            "body argument #" + i + " must have the type of result #" + i)
                    || !check(
                            shapeEq(b, r) && sameKindIgnoringFp(b.elementType(), r.elementType()),
                            c,
                            "body argument #" + (n + i) + " must have the type of result #" + i)) {
                return null;
            }
            TensorType init = i < inits.size() ? inits.get(i) : null;
            if (init != null
                    && (!check(
                                    shapeEq(init, r),
                                    c,
                                    "body result #" + i + " must have the init value shape")
                            || !check(
                                    promotableIgnoringFp(init.elementType(), r.elementType()),
                                    c,
                                    "init value #"
                                            + i
                                            + " does not promote to the body result type"))) {
                return null;
            }
            if (!check(
                    promotableIgnoringFp(inputs.get(i), b.elementType()),
                    c,
                    "element type "
                            + inputs.get(i).mlir()
                            + " does not promote to "
                            + b.elementType().mlir())) {
                return null;
            }
            if (!check(
                    b.rank() <= allowed.length,
                    c,
                    "body argument #" + (n + i) + " has rank > " + allowed.length)) {
                return null;
            }
            int k = 0;
            for (int d = 0; d < allowed.length && k < b.rank(); d++) {
                if (dimEq(allowed[d], b.dim(k))) {
                    k++;
                }
            }
            if (!check(
                    k == b.rank(),
                    c,
                    "body argument #" + (n + i) + " shape is incompatible with the input")) {
                return null;
            }
            es.add(r.elementType());
        }
        return es;
    }

    private static boolean sameKindIgnoringFp(ElementType a, ElementType b) {
        return a instanceof FloatType && b instanceof FloatType || elemEq(a, b);
    }

    /** {@code isPromotableElementType(a, b, ignoreFpPrecision=true)} of the reference. */
    private static boolean promotableIgnoringFp(ElementType a, ElementType b) {
        boolean sameKind =
                a instanceof IntegerType && b instanceof IntegerType
                        || a instanceof FloatType && b instanceof FloatType
                        || a instanceof ComplexType && b instanceof ComplexType
                        || a instanceof QuantizedType && b instanceof QuantizedType;
        if (!sameKind) {
            return false;
        }
        if (a instanceof QuantizedType qa && b instanceof QuantizedType qb) {
            return qa.storageType().bitWidth() <= qb.storageType().bitWidth()
                    && qa.expressedType().equals(qb.expressedType());
        }
        return a instanceof FloatType || bits(a) <= bits(b);
    }

    private static List<ElementType> elems(List<Value> values) {
        return values.stream().map(v -> tt(v) == null ? null : tt(v).elementType()).toList();
    }

    private static List<TensorType> types(List<Value> values) {
        return values.stream().map(Verifier::tt).toList();
    }

    private static TensorType scalar(ElementType e) {
        return TensorType.of(e);
    }

    // ================================================================== quantization

    private static QuantizedType q(TensorType t) {
        return t != null && t.elementType() instanceof QuantizedType qt ? qt : null;
    }

    /** Same quantized type except possibly the quantization dimension. */
    private static boolean sameExceptAxis(QuantizedType a, QuantizedType b) {
        return a.storageType().equals(b.storageType())
                && java.util.Objects.equals(a.storageMin(), b.storageMin())
                && java.util.Objects.equals(a.storageMax(), b.storageMax())
                && a.expressedType().equals(b.expressedType())
                && a.scales().equals(b.scales())
                && a.zeroPoints().equals(b.zeroPoints());
    }

    /**
     * Element types of a shape-changing op (reshape, transpose, broadcasts): equal; for per-axis
     * quantization the quantization dimension may differ. Returns both types when per-axis.
     */
    private boolean checkShapeOpElements(
            TensorType x, TensorType r, String c, boolean allowScaleChange) {
        QuantizedType qx = q(x);
        QuantizedType qr = q(r);
        if (qx == null || qr == null) {
            check(
                    elemEq(x.elementType(), r.elementType()),
                    c,
                    "operand and result element types must match");
            return false;
        }
        if (!qx.isPerAxis() || !qr.isPerAxis()) {
            check(qx.equals(qr), c, "expect same quantization scale and zero_point");
            return false;
        }
        if (!allowScaleChange) {
            check(sameExceptAxis(qx, qr), c, "expect same quantization scales and zero_points");
        }
        return true;
    }

    /** C2..C7 of {@code add}: quantized operands and result agree. */
    private void checkQuantizedElementwise(Operation op) {
        List<TensorType> all = new ArrayList<>();
        op.operands().forEach(v -> all.add(tt(v)));
        op.results().forEach(v -> all.add(tt(v)));
        if (all.stream().anyMatch(java.util.Objects::isNull)
                || all.stream().noneMatch(Verifier::isQuantized)) {
            return;
        }
        if (!check(
                all.stream().allMatch(Verifier::isQuantized),
                "C2",
                "all operands and results must be either quantized or non-quantized")) {
            return;
        }
        QuantizedType first = q(all.getFirst());
        for (TensorType t : all) {
            check(
                    q(t).storageType().equals(first.storageType()),
                    "C3",
                    "mismatched operands and result quantization storage types");
            check(
                    q(t).expressedType().equals(first.expressedType()),
                    "C4",
                    "mismatched operands and result quantization expressed types");
        }
        QuantizedType result = q(all.getLast());
        boolean anyAxis = false;
        for (int i = 0; i < all.size() - 1; i++) {
            QuantizedType operand = q(all.get(i));
            anyAxis |= operand.isPerAxis();
            if (operand.isPerAxis()) {
                check(
                        java.util.Objects.equals(
                                operand.quantizationDimension(), result.quantizationDimension()),
                        i == 0 ? "C6" : "C7",
                        "quantization_dimension of operand #" + i + " and result are not same");
            }
        }
        check(
                anyAxis == result.isPerAxis(),
                "C5",
                "result is per-axis quantized if and only if an operand is");
    }

    /**
     * Quantization constraints shared by dot_general (C14..C20) and convolution (C28..C34). {@code
     * rhsContracting}: dimensions the rhs quantization axis may not be in; {@code kernelOutput},
     * {@code resultFeature}: axes the per-axis quantization must use (or -1).
     */
    private void checkQuantizedContraction(
            TensorType l,
            TensorType rh,
            TensorType r,
            long[] rhsContracting,
            long kernelOutput,
            long resultFeature) {
        if (!isQuantized(l) && !isQuantized(rh) && !isQuantized(r)) {
            return;
        }
        if (!check(
                isQuantized(rh) && isQuantized(l) == isQuantized(r),
                "quant",
                "rhs should be quantized for quantized operations and"
                        + " is_quantized(lhs)=is_quantized(result) should hold")) {
            return;
        }
        QuantizedType qr = q(rh);
        boolean dot = kernelOutput < 0;
        if (dot) {
            check(
                    qr.zeroPoints().stream().allMatch(z -> z == 0),
                    "C15",
                    "zero points of rhs should be 0");
        }
        if (qr.isPerAxis()) {
            check(
                    !dot || !contains(rhsContracting, qr.quantizationDimension()),
                    "C16",
                    "quantization dimension of rhs should not be in the contracting dimension of"
                            + " rhs");
            if (kernelOutput >= 0) {
                check(
                        qr.quantizationDimension() == kernelOutput,
                        "quant",
                        "quantization dimension of rhs should be same with"
                                + " kernel_output_feature_dimension");
            }
        }
        if (isQuantized(l)) {
            QuantizedType ql = q(l);
            QuantizedType qo = q(r);
            check(
                    ql.storageType().equals(qr.storageType()),
                    "quant",
                    "mismatched lhs and rhs quantization storage types");
            check(
                    ql.expressedType().equals(qr.expressedType())
                            && qr.expressedType().equals(qo.expressedType()),
                    "quant",
                    "mismatched lhs, rhs and result quantization expressed types");
            check(
                    qr.isPerAxis() || !qo.isPerAxis(),
                    "quant",
                    "mismatched rhs and result quantization granularity");
            if (qo.isPerAxis() && resultFeature >= 0) {
                check(
                        qo.quantizationDimension() == resultFeature,
                        "quant",
                        "quantization dimension of result should be same with"
                                + " output_feature_dimension");
            }
        } else {
            check(
                    l.elementType().equals(qr.expressedType())
                            && r.elementType().equals(l.elementType()),
                    "quant",
                    "mismatched rhs quantization expressed type and lhs and result element type");
        }
    }

    // ================================================================== dynamic shape operands

    /** The values of a constant integer operand, or null. */
    private static long[] constant(Value v) {
        if (v.definingOp() != null
                && v.definingOp().asOp() instanceof ConstantOp c
                && c.value() instanceof DenseElementsAttr d
                && (d.type().elementType() instanceof IntegerType
                        || d.type().elementType() instanceof IndexType)) {
            long[] out = new long[(int) d.numElements()];
            for (int i = 0; i < out.length; i++) {
                out[i] = d.getLong(i);
            }
            return out;
        }
        return null;
    }

    /** A shape operand: a static 1-D integer tensor; its constant value must match the result. */
    private long[] checkShapeOperand(
            Value shape, TensorType result, String c, boolean mustBeStatic) {
        TensorType s = tt(shape);
        if (s == null) {
            return null;
        }
        if (mustBeStatic
                && !check(s.hasStaticShape(), c, "shape operand must be statically shaped")) {
            return null;
        }
        long[] values = constant(shape);
        if (values != null && result != null && values.length == result.rank()) {
            boolean ok = true;
            for (int i = 0; i < values.length; i++) {
                ok &= result.dim(i) == DYN || result.dim(i) == values[i];
            }
            check(
                    ok,
                    c,
                    "output shape "
                            + str(values)
                            + " is incompatible with the result type "
                            + result.mlir());
        }
        return values;
    }

    // ================================================================== functions

    @Override
    public Void visitFuncReturn(FuncReturnOp op) {
        Operation parent = findParentFunc(op.operation());
        if (parent != null) {
            FunctionType ft = (FunctionType) new FuncOp(parent).functionType();
            check(
                    typesEq(op.operation().operandTypes(), ft.results()),
                    "func",
                    "returned types "
                            + op.operation().operandTypes()
                            + " do not match the function results "
                            + ft.results());
        }
        return null;
    }

    private Operation findParentFunc(Operation ret) {
        for (Operation f : program.operations()) {
            if (f.name().equals("func.func") && !f.region(0).isEmpty()) {
                if (f.region(0).blocks().getLast().terminator() == ret) {
                    return f;
                }
            }
        }
        return null;
    }

    @Override
    public Void visitFunc(FuncOp op) {
        if (!op.body().isEmpty()) {
            FunctionType ft = (FunctionType) op.functionType();
            check(
                    typesEq(argumentTypes(op.body()), ft.inputs()),
                    "func",
                    "block arguments do not match the function inputs");
        }
        return null;
    }

    @Override
    public Void visitCall(CallOp op) {
        var callee = program.function(op.callee());
        if (!check(callee.isPresent(), "call", "unknown function @" + op.callee())) {
            return null;
        }
        FunctionType ft = (FunctionType) callee.get().functionType();
        check(
                typesEq(op.operation().operandTypes(), ft.inputs()),
                "call",
                "operand types do not match @" + op.callee());
        check(
                typesEq(op.operation().resultTypes(), ft.results()),
                "call",
                "result types do not match @" + op.callee());
        return null;
    }

    @Override
    public Void visitComposite(CompositeOp op) {
        check(
                op.compositeName().matches("[a-zA-Z][a-zA-Z0-9_]*([.][a-zA-Z0-9_$]+)+"),
                "C1",
                "name '" + op.compositeName() + "' is not a namespaced op name");
        var callee = program.function(op.decomposition());
        if (!check(callee.isPresent(), "C2", "unknown decomposition @" + op.decomposition())) {
            return null;
        }
        FunctionType ft = (FunctionType) callee.get().functionType();
        check(
                typesEq(op.operation().operandTypes(), ft.inputs()),
                "C3",
                "input types do not match the decomposition");
        check(
                typesEq(op.operation().resultTypes(), ft.results()),
                "C4",
                "result types do not match the decomposition");
        return null;
    }

    // ================================================================== elementwise

    /** {@code type(operands...) = type(result)} for non-quantized tensors (shapes always). */
    private void sameTypes(Operation op, String c) {
        List<Value> all = new ArrayList<>(op.operands());
        all.addAll(op.results());
        TensorType first = null;
        boolean quantized = false;
        for (Value v : all) {
            quantized |= isQuantized(tt(v));
        }
        for (Value v : all) {
            TensorType t = tt(v);
            if (t == null) {
                continue;
            }
            if (first == null) {
                first = t;
                continue;
            }
            if (!check(shapeEq(first, t), c, "operands and result must have the same shape")) {
                return;
            }
            if (!quantized
                    && !check(
                            elemEq(first.elementType(), t.elementType()),
                            c,
                            "operands and result must have the same element type")) {
                return;
            }
        }
    }

    private Void unarySame(Op op) {
        sameTypes(op.operation(), "C1");
        return null;
    }

    private Void binarySame(Op op) {
        sameTypes(op.operation(), "C1");
        return null;
    }

    @Override
    public Void visitNegate(NegateOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitExponential(ExponentialOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitExponentialMinusOne(ExponentialMinusOneOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitExp2(Exp2Op op) {
        return unarySame(op);
    }

    @Override
    public Void visitLog(LogOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitLogPlusOne(LogPlusOneOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitLog2(Log2Op op) {
        return unarySame(op);
    }

    @Override
    public Void visitLogistic(LogisticOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitSqrt(SqrtOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitRsqrt(RsqrtOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitCbrt(CbrtOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitSine(SineOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitCosine(CosineOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitTan(TanOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitTanh(TanhOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitCeil(CeilOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitFloor(FloorOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitRoundNearestAfz(RoundNearestAfzOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitRoundNearestEven(RoundNearestEvenOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitSign(SignOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitNot(NotOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitPopcnt(PopcntOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitCountLeadingZeros(CountLeadingZerosOp op) {
        return unarySame(op);
    }

    @Override
    public Void visitAdd(AddOp op) {
        checkQuantizedElementwise(op.operation());
        return binarySame(op);
    }

    @Override
    public Void visitSubtract(SubtractOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitMultiply(MultiplyOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitDivide(DivideOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitRemainder(RemainderOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitPower(PowerOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitMaximum(MaximumOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitMinimum(MinimumOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitAtan2(Atan2Op op) {
        return binarySame(op);
    }

    @Override
    public Void visitAnd(AndOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitOr(OrOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitXor(XorOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitShiftLeft(ShiftLeftOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitShiftRightArithmetic(ShiftRightArithmeticOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitShiftRightLogical(ShiftRightLogicalOp op) {
        return binarySame(op);
    }

    @Override
    public Void visitAbs(AbsOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
        ElementType want =
                x.elementType() instanceof ComplexType c ? c.elementType() : x.elementType();
        if (!isQuantized(x)) {
            check(
                    elemEq(want, r.elementType()),
                    "C2",
                    "result element type must be " + want.mlir());
        }
        return null;
    }

    private Void realImag(Value operand, Value result) {
        TensorType x = tt(operand);
        TensorType r = tt(result);
        if (x == null || r == null) {
            return null;
        }
        check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
        ElementType want =
                x.elementType() instanceof ComplexType c ? c.elementType() : x.elementType();
        check(elemEq(want, r.elementType()), "C2", "result element type must be " + want.mlir());
        return null;
    }

    @Override
    public Void visitReal(RealOp op) {
        return realImag(op.operand(), op.result());
    }

    @Override
    public Void visitImag(ImagOp op) {
        return realImag(op.operand(), op.result());
    }

    @Override
    public Void visitComplex(ComplexOp op) {
        TensorType l = tt(op.lhs());
        TensorType rh = tt(op.rhs());
        TensorType r = tt(op.result());
        if (l == null || rh == null || r == null) {
            return null;
        }
        check(typeEq(l, rh), "C1", "lhs and rhs must have the same type");
        check(shapeEq(r, l), "C2", "result shape must equal the operand shape");
        check(
                r.elementType() instanceof ComplexType c && c.elementType().equals(l.elementType()),
                "C3",
                "result element type must be complex<" + l.elementType().mlir() + ">");
        return null;
    }

    @Override
    public Void visitIsFinite(IsFiniteOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x != null && r != null) {
            check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
            check(isBool(r.elementType()), "C1", "result element type must be i1");
        }
        return null;
    }

    @Override
    public Void visitCompare(CompareOp op) {
        TensorType l = tt(op.lhs());
        TensorType rh = tt(op.rhs());
        TensorType r = tt(op.result());
        if (l == null || rh == null || r == null) {
            return null;
        }
        check(
                elemEq(l.elementType(), rh.elementType()),
                "C1",
                "lhs and rhs must have the same element type");
        check(
                shapeEq(l, rh) && shapeEq(l, r),
                "C2",
                "lhs, rhs and result must have the same shape");
        check(isBool(r.elementType()), "C2", "result element type must be i1");
        op.compareType()
                .ifPresent(
                        ct -> {
                            // reference implementation: only WEAKORDER is restricted (to floats)
                            check(
                                    ct != ComparisonType.WEAKORDER
                                            || l.elementType() instanceof FloatType,
                                    "C3",
                                    "WEAKORDER comparison type needs floating-point elements");
                        });
        return null;
    }

    @Override
    public Void visitSelect(SelectOp op) {
        TensorType p = tt(op.pred());
        TensorType t = tt(op.onTrue());
        TensorType f = tt(op.onFalse());
        TensorType r = tt(op.result());
        if (p == null || t == null || f == null || r == null) {
            return null;
        }
        check(isBool(p.elementType()), "C1", "pred must have element type i1");
        check(
                p.rank() == 0 || shapeEq(p, t),
                "C1",
                "pred must be rank 0 or have the shape of on_true");
        check(
                typeEq(t, f) && typeEq(t, r),
                "C2",
                "on_true, on_false and result must have the same type");
        return null;
    }

    @Override
    public Void visitClamp(ClampOp op) {
        TensorType mn = tt(op.min());
        TensorType x = tt(op.operand());
        TensorType mx = tt(op.max());
        TensorType r = tt(op.result());
        if (mn == null || x == null || mx == null || r == null) {
            return null;
        }
        check(
                mn.rank() == 0 || shapeEq(mn, x),
                "C1",
                "min must be rank 0 or have the shape of operand");
        check(
                mx.rank() == 0 || shapeEq(mx, x),
                "C2",
                "max must be rank 0 or have the shape of operand");
        check(
                elemEq(mn.elementType(), x.elementType())
                        && elemEq(mx.elementType(), x.elementType()),
                "C3",
                "min, operand and max must have the same element type");
        check(typeEq(x, r), "C4", "result must have the operand type");
        return null;
    }

    @Override
    public Void visitConvert(ConvertOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x != null && r != null) {
            check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
        }
        return null;
    }

    @Override
    public Void visitBitcastConvert(BitcastConvertOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        int e = bits(x.elementType());
        int e2 = bits(r.elementType());
        int rank = x.rank();
        if (e2 == e) {
            check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
        } else if (e2 < e) {
            boolean ok = r.rank() == rank + 1 && dimEq(r.dim(rank), e / e2);
            for (int i = 0; ok && i < rank; i++) {
                ok = dimEq(r.dim(i), x.dim(i));
            }
            check(ok, "C1", "result shape must be the operand shape followed by " + (e / e2));
        } else {
            boolean ok = rank >= 1 && r.rank() == rank - 1 && dimEq(x.dim(rank - 1), e2 / e);
            for (int i = 0; ok && i < rank - 1; i++) {
                ok = dimEq(r.dim(i), x.dim(i));
            }
            check(ok, "C1", "operand shape must be the result shape followed by " + (e2 / e));
        }
        check(
                (x.elementType() instanceof ComplexType)
                        == (r.elementType() instanceof ComplexType),
                "C2",
                "complex types can only be bitcast to complex types");
        return null;
    }

    @Override
    public Void visitReducePrecision(ReducePrecisionOp op) {
        sameTypes(op.operation(), "C1");
        check(op.exponentBits() >= 1, "C2", "exponent_bits must be >= 1");
        check(op.mantissaBits() >= 0, "C3", "mantissa_bits must be >= 0");
        return null;
    }

    @Override
    public Void visitUniformDequantize(UniformDequantizeOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x != null && r != null) {
            check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
            if (x.elementType() instanceof QuantizedType q) {
                check(
                        r.elementType().equals(q.expressedType()),
                        "C2",
                        "result element type must be the expressed type");
            }
        }
        return null;
    }

    @Override
    public Void visitConstant(ConstantOp op) {
        if (op.value() instanceof DenseElementsAttr d && tt(op.result()) instanceof TensorType r) {
            ElementType want =
                    r.elementType() instanceof QuantizedType q ? q.storageType() : r.elementType();
            check(
                    shapeEq(d.type(), r) && elemEq(d.type().elementType(), want),
                    "C1",
                    "value type "
                            + d.type().mlir()
                            + " does not match the result type "
                            + r.mlir());
        }
        return null;
    }

    @Override
    public Void visitIota(IotaOp op) {
        TensorType r = tt(op.result());
        if (r != null) {
            check(
                    op.iotaDimension() >= 0 && op.iotaDimension() < r.rank(),
                    "C1",
                    "iota_dimension "
                            + op.iotaDimension()
                            + " is out of range for rank "
                            + r.rank());
        }
        return null;
    }

    @Override
    public Void visitDynamicIota(DynamicIotaOp op) {
        TensorType r = tt(op.result());
        TensorType s = tt(op.outputShape());
        if (r != null) {
            check(
                    op.iotaDimension() >= 0 && op.iotaDimension() < r.rank(),
                    "C1",
                    "iota_dimension " + op.iotaDimension() + " is out of range");
        }
        if (r != null && s != null && s.rank() == 1) {
            check(dimEq(s.dim(0), r.rank()), "C2", "result rank must equal size(output_shape)");
        }
        checkShapeOperand(op.outputShape(), r, "C2", false);
        return null;
    }

    @Override
    public Void visitGetDimensionSize(GetDimensionSizeOp op) {
        TensorType x = tt(op.operand());
        if (x != null) {
            check(op.dimension() >= 0 && op.dimension() < x.rank(), "C1", "dimension out of range");
        }
        return null;
    }

    @Override
    public Void visitSetDimensionSize(SetDimensionSizeOp op) {
        TensorType x = tt(op.operand());
        if (x != null) {
            check(op.dimension() >= 0 && op.dimension() < x.rank(), "C1", "dimension out of range");
        }
        return null;
    }

    @Override
    public Void visitReshape(ReshapeOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        check(
                r.hasStaticShape() || r.encoding() != null,
                "type",
                "result must be statically shaped or bounded");
        if (x.hasStaticShape() && r.hasStaticShape()) {
            check(
                    x.numElements() == r.numElements(),
                    "C2",
                    "operand has " + x.numElements() + " elements, result " + r.numElements());
        }
        if (checkShapeOpElements(x, r, "C1", false) && x.hasStaticShape() && r.hasStaticShape()) {
            int ax = q(x).quantizationDimension();
            int ar = q(r).quantizationDimension();
            if (ax < x.rank() && ar < r.rank()) {
                check(
                        product(x, 0, ax) == product(r, 0, ar),
                        "C3",
                        "product of dimensions before quantization dimension must match between"
                                + " operand and result");
                check(
                        x.dim(ax) == r.dim(ar),
                        "C3",
                        "expect same quantization dimension size for operand and result");
                check(
                        product(x, ax + 1, x.rank()) == product(r, ar + 1, r.rank()),
                        "C3",
                        "product of dimensions after quantization dimension must match between"
                                + " operand and result");
            }
        }
        return null;
    }

    private static long product(TensorType t, int from, int to) {
        long p = 1;
        for (int i = from; i < to; i++) {
            p *= t.dim(i);
        }
        return p;
    }

    @Override
    public Void visitDynamicReshape(DynamicReshapeOp op) {
        TensorType r = tt(op.result());
        TensorType s = tt(op.outputShape());
        if (r != null && s != null && s.rank() == 1) {
            check(dimEq(s.dim(0), r.rank()), "C4", "size(output_shape) must equal the result rank");
        }
        long[] shape = checkShapeOperand(op.outputShape(), r, "C4", true);
        TensorType x = tt(op.operand());
        if (shape != null
                && x != null
                && x.hasStaticShape()
                && Arrays.stream(shape).allMatch(v -> v >= 0)) {
            long n = Arrays.stream(shape).reduce(1, (a, b) -> a * b);
            check(
                    n == x.numElements(),
                    "C2",
                    "input has " + x.numElements() + " elements, but output_shape has " + n);
        }
        return null;
    }

    @Override
    public Void visitTranspose(TransposeOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        long[] perm = op.permutation();
        if (x == null || r == null) {
            return null;
        }
        if (!check(
                perm.length == x.rank() && unique(perm) && inRange(perm, 0, x.rank()),
                "C2",
                "permutation " + str(perm) + " is not a permutation of range(" + x.rank() + ")")) {
            return null;
        }
        boolean ok = r.rank() == perm.length;
        for (int i = 0; ok && i < perm.length; i++) {
            ok = dimEq(r.dim(i), x.dim((int) perm[i]));
        }
        check(ok, "C3", "result shape must be the permuted operand shape");
        if (checkShapeOpElements(x, r, "C1", false)) {
            int axis = q(r).quantizationDimension();
            if (axis < perm.length) {
                check(
                        q(x).quantizationDimension() == perm[axis],
                        "C4",
                        "operand quantization_dimension "
                                + q(x).quantizationDimension()
                                + " is not same as permutation["
                                + axis
                                + "] = "
                                + perm[axis]);
            }
        }
        return null;
    }

    @Override
    public Void visitReverse(ReverseOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        check(typeEq(x, r), "C1", "result must have the operand type");
        check(unique(op.dimensions()), "C2", "dimensions must be unique");
        check(inRange(op.dimensions(), 0, r.rank()), "C3", "dimensions out of range");
        return null;
    }

    private void checkBroadcastDims(TensorType x, TensorType r, long[] dims) {
        if (!check(
                dims.length == x.rank(),
                "C2",
                "size(broadcast_dimensions) must equal the operand rank " + x.rank())) {
            return;
        }
        if (!check(inRange(dims, 0, r.rank()), "C3", "broadcast_dimensions out of range")) {
            return;
        }
        if (!check(unique(dims), "C4", "broadcast_dimensions must be unique")) {
            return;
        }
        for (int d = 0; d < dims.length; d++) {
            long od = x.dim(d);
            long rd = r.dim((int) dims[d]);
            if (!check(
                    od == 1 || dimEq(od, rd),
                    "C5",
                    "operand dimension "
                            + d
                            + " ("
                            + od
                            + ") is not 1 and differs from result dimension "
                            + dims[d]
                            + " ("
                            + rd
                            + ")")) {
                return;
            }
        }
    }

    @Override
    public Void visitBroadcastInDim(BroadcastInDimOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        check(
                r.hasStaticShape() || r.encoding() != null,
                "type",
                "result must be statically shaped or bounded");
        checkBroadcastQuantization(x, r, op.broadcastDimensions());
        checkBroadcastDims(x, r, op.broadcastDimensions());
        return null;
    }

    /** C1 and C6 of broadcast_in_dim / dynamic_broadcast_in_dim. */
    private void checkBroadcastQuantization(TensorType x, TensorType r, long[] dims) {
        if (!checkShapeOpElements(x, r, "C1", true)) {
            return;
        }
        QuantizedType qx = q(x);
        QuantizedType qr = q(r);
        int axis = qx.quantizationDimension();
        if (axis < dims.length) {
            check(
                    qr.quantizationDimension() == dims[axis],
                    "C6",
                    "result quantization_dimension "
                            + qr.quantizationDimension()
                            + " not same as broadcast_dimensions["
                            + axis
                            + "] = "
                            + dims[axis]);
        }
        boolean single = axis < x.rank() && x.dim(axis) == 1;
        for (int i = 0; i < qr.scales().size(); i++) {
            int j = single ? 0 : i;
            if (j >= qx.scales().size()) {
                break;
            }
            if (!check(
                            qr.scales().get(i).equals(qx.scales().get(j)),
                            "C6",
                            "mismatch result scale " + i)
                    || !check(
                            qr.zeroPoints().get(i).equals(qx.zeroPoints().get(j)),
                            "C6",
                            "mismatch result zero_point " + i)) {
                return;
            }
        }
    }

    @Override
    public Void visitDynamicBroadcastInDim(DynamicBroadcastInDimOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        TensorType s = tt(op.outputDimensions());
        if (x == null || r == null) {
            return null;
        }
        checkBroadcastQuantization(x, r, op.broadcastDimensions());
        checkBroadcastDims(x, r, op.broadcastDimensions());
        long[] shape = checkShapeOperand(op.outputDimensions(), r, "C7", true);
        long[] bd = op.broadcastDimensions();
        if (shape != null && bd.length == x.rank() && inRange(bd, 0, shape.length)) {
            for (int d = 0; d < bd.length; d++) {
                long v = shape[(int) bd[d]];
                if (!check(
                        x.dim(d) == DYN || x.dim(d) == 1 || v < 0 || x.dim(d) == v,
                        "C5",
                        "size of operand dimension "
                                + d
                                + " ("
                                + x.dim(d)
                                + ") is not equal to 1 or value of shape at index "
                                + bd[d]
                                + " ("
                                + v
                                + ")")) {
                    break;
                }
            }
        }
        if (s != null && s.rank() == 1) {
            check(
                    dimEq(s.dim(0), r.rank()),
                    "C7",
                    "size(output_dimensions) must equal the result rank");
        }
        long[] exp = op.knownExpandingDimensions().orElse(new long[0]);
        long[] nonexp = op.knownNonexpandingDimensions().orElse(new long[0]);
        check(unique(concat(exp, nonexp)), "C8", "known (non)expanding dimensions must be unique");
        check(inRange(exp, 0, x.rank()), "C9", "known_expanding_dimensions out of range");
        check(inRange(nonexp, 0, x.rank()), "C10", "known_nonexpanding_dimensions out of range");
        return null;
    }

    @Override
    public Void visitBroadcast(BroadcastOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        long[] sizes = op.broadcastSizes();
        boolean ok = r.rank() == sizes.length + x.rank();
        for (int i = 0; ok && i < sizes.length; i++) {
            ok = dimEq(r.dim(i), sizes[i]);
        }
        for (int i = 0; ok && i < x.rank(); i++) {
            ok = dimEq(r.dim(sizes.length + i), x.dim(i));
        }
        check(ok, "C1", "result shape must be broadcast_sizes followed by the operand shape");
        return null;
    }

    @Override
    public Void visitSlice(SliceOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        long[] s = op.startIndices();
        long[] l = op.limitIndices();
        long[] st = op.strides();
        check(elemEq(x.elementType(), r.elementType()), "C1", "element types must match");
        if (!check(
                s.length == x.rank() && l.length == x.rank() && st.length == x.rank(),
                "C2",
                "start_indices, limit_indices and strides must have " + x.rank() + " elements")) {
            return null;
        }
        boolean ok = true;
        for (int i = 0; i < s.length; i++) {
            ok &= 0 <= s[i] && s[i] <= l[i] && (x.dim(i) == DYN || l[i] <= x.dim(i));
        }
        if (!check(ok, "C3", "need 0 <= start_indices <= limit_indices <= shape(operand)")) {
            return null;
        }
        if (!check(Arrays.stream(st).allMatch(v -> v > 0), "C4", "strides must be positive")) {
            return null;
        }
        boolean shape = r.rank() == s.length;
        for (int i = 0; shape && i < s.length; i++) {
            shape = dimEq(r.dim(i), (l[i] - s[i] + st[i] - 1) / st[i]);
        }
        check(shape, "C5", "result shape must be ceil((limit_indices - start_indices) / strides)");
        return null;
    }

    @Override
    public Void visitDynamicSlice(DynamicSliceOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        long[] sizes = op.sliceSizes();
        check(elemEq(x.elementType(), r.elementType()), "C1", "element types must match");
        if (!check(
                op.startIndices().size() == x.rank() && sizes.length == x.rank(),
                "C2",
                "need " + x.rank() + " start indices and slice sizes")) {
            return null;
        }
        checkSameIndexTypes(op.startIndices(), "C3");
        boolean ok = true;
        for (int i = 0; i < sizes.length; i++) {
            ok &= sizes[i] >= 0 && (x.dim(i) == DYN || sizes[i] <= x.dim(i));
        }
        check(ok, "C4", "need 0 <= slice_sizes <= shape(operand)");
        boolean shape = r.rank() == sizes.length;
        for (int i = 0; shape && i < sizes.length; i++) {
            shape = dimEq(r.dim(i), sizes[i]);
        }
        check(shape, "C5", "result shape must equal slice_sizes");
        return null;
    }

    private void checkSameIndexTypes(List<Value> indices, String c) {
        for (Value v : indices) {
            TensorType t = tt(v);
            if (t != null
                    && !check(
                            t.rank() == 0 && isInteger(t.elementType()),
                            c,
                            "start indices must be rank-0 integer tensors")) {
                return;
            }
        }
        for (int i = 1; i < indices.size(); i++) {
            if (!check(
                    typeEq(indices.get(0).type(), indices.get(i).type()),
                    c,
                    "start indices must have the same type")) {
                return;
            }
        }
    }

    @Override
    public Void visitDynamicUpdateSlice(DynamicUpdateSliceOp op) {
        TensorType x = tt(op.operand());
        TensorType u = tt(op.update());
        TensorType r = tt(op.result());
        if (x == null || u == null || r == null) {
            return null;
        }
        check(typeEq(x, r), "C1", "result must have the operand type");
        check(
                elemEq(u.elementType(), x.elementType()),
                "C2",
                "update must have the operand element type");
        if (!check(u.rank() == x.rank(), "C3", "update must have the operand rank")) {
            return null;
        }
        check(op.startIndices().size() == x.rank(), "C4", "need " + x.rank() + " start indices");
        checkSameIndexTypes(op.startIndices(), "C5");
        boolean ok = true;
        for (int i = 0; i < u.rank(); i++) {
            ok &= u.dim(i) == DYN || x.dim(i) == DYN || u.dim(i) <= x.dim(i);
        }
        check(ok, "C6", "need shape(update) <= shape(operand)");
        return null;
    }

    @Override
    public Void visitConcatenate(ConcatenateOp op) {
        List<Value> in = op.inputs();
        TensorType r = tt(op.result());
        if (!check(!in.isEmpty(), "C3", "needs at least one input") || r == null) {
            return null;
        }
        TensorType first = tt(in.getFirst());
        if (first == null) {
            return null;
        }
        long dim = op.dimension();
        if (!check(dim >= 0 && dim < first.rank(), "C4", "dimension " + dim + " out of range")) {
            return null;
        }
        long total = 0;
        boolean dynamic = false;
        for (Value v : in) {
            TensorType t = tt(v);
            if (t == null) {
                continue;
            }
            if (!check(
                            elemEq(t.elementType(), first.elementType()),
                            "C1",
                            "inputs must have the same element type")
                    || !check(t.rank() == first.rank(), "C2", "inputs must have the same rank")) {
                return null;
            }
            for (int i = 0; i < t.rank(); i++) {
                if (i != dim
                        && !check(
                                dimEq(t.dim(i), first.dim(i)),
                                "C2",
                                "inputs differ in dimension " + i)) {
                    return null;
                }
            }
            if (t.dim((int) dim) == DYN) {
                dynamic = true;
            } else {
                total += t.dim((int) dim);
            }
        }
        check(
                elemEq(r.elementType(), first.elementType()),
                "C5",
                "result element type must match the inputs");
        boolean shape = r.rank() == first.rank();
        for (int i = 0; shape && i < r.rank(); i++) {
            shape = i == dim ? (dynamic || dimEq(r.dim(i), total)) : dimEq(r.dim(i), first.dim(i));
        }
        check(shape, "C6", "result shape must be the inputs concatenated along dimension " + dim);
        return null;
    }

    @Override
    public Void visitPad(PadOp op) {
        TensorType x = tt(op.operand());
        TensorType p = tt(op.paddingValue());
        TensorType r = tt(op.result());
        if (x == null || p == null || r == null) {
            return null;
        }
        check(
                elemEq(x.elementType(), p.elementType())
                        && elemEq(x.elementType(), r.elementType()),
                "C1",
                "operand, padding_value and result must have the same element type");
        long[] lo = op.edgePaddingLow();
        long[] hi = op.edgePaddingHigh();
        long[] in = op.interiorPadding();
        if (!check(
                lo.length == x.rank() && hi.length == x.rank() && in.length == x.rank(),
                "C2",
                "padding attributes must have " + x.rank() + " elements")) {
            return null;
        }
        if (!check(
                Arrays.stream(in).allMatch(v -> v >= 0), "C3", "interior_padding must be >= 0")) {
            return null;
        }
        boolean shape = r.rank() == x.rank();
        for (int i = 0; shape && i < x.rank(); i++) {
            if (x.dim(i) != DYN) {
                shape =
                        dimEq(
                                r.dim(i),
                                x.dim(i) + lo[i] + Math.max(x.dim(i) - 1, 0) * in[i] + hi[i]);
            }
        }
        check(shape, "C4", "result shape does not match the padding");
        return null;
    }

    @Override
    public Void visitGetTupleElement(GetTupleElementOp op) {
        if (op.operand().type() instanceof TupleType t) {
            if (check(
                    op.index() >= 0 && op.index() < t.types().size(), "C1", "index out of range")) {
                check(
                        typeEq(op.result().type(), t.types().get((int) op.index())),
                        "C2",
                        "result type must be the tuple element type");
            }
        }
        return null;
    }

    @Override
    public Void visitTuple(TupleOp op) {
        if (op.result().type() instanceof TupleType t) {
            check(
                    typesEq(op.operation().operandTypes(), t.types()),
                    "C1",
                    "result must be tuple of the operand types");
        }
        return null;
    }

    @Override
    public Void visitOptimizationBarrier(OptimizationBarrierOp op) {
        check(
                typesEq(op.operation().operandTypes(), op.operation().resultTypes()),
                "C1",
                "results must have the operand types");
        return null;
    }

    // ================================================================== gather / scatter

    @Override
    public Void visitGather(com.qxotic.jota.stablehlo.op.GatherOp op) {
        checkGather(
                op.dimensionNumbers(),
                tt(op.operand()),
                tt(op.startIndices()),
                op.sliceSizes(),
                tt(op.result()),
                true);
        return null;
    }

    @Override
    public Void visitDynamicGather(DynamicGatherOp op) {
        long[] sizes = null;
        if (op.sliceSizes().definingOp() != null
                && op.sliceSizes().definingOp().asOp() instanceof ConstantOp c
                && c.value() instanceof DenseElementsAttr d) {
            sizes = new long[(int) d.numElements()];
            for (int i = 0; i < sizes.length; i++) {
                sizes[i] = d.getLong(i);
            }
        }
        checkGather(
                op.dimensionNumbers(),
                tt(op.operand()),
                tt(op.startIndices()),
                sizes,
                tt(op.result()),
                false);
        return null;
    }

    private void checkGather(
            GatherDimensionNumbers dn,
            TensorType x,
            TensorType s,
            long[] sizes,
            TensorType r,
            boolean batching) {
        if (x == null || s == null || r == null) {
            return;
        }
        long[] off = dn.offsetDims();
        long[] col = dn.collapsedSliceDims();
        long[] ob = dn.operandBatchingDims();
        long[] sb = dn.startIndicesBatchingDims();
        long[] map = dn.startIndexMap();
        long ivd = dn.indexVectorDim();
        int n = x.rank();
        if (!check(
                n == off.length + col.length + ob.length,
                "C1",
                "rank(operand) must equal size(offset_dims) + size(collapsed_slice_dims) +"
                        + " size(operand_batching_dims)")) {
            return;
        }
        if (!check(
                ivd >= 0 && ivd <= s.rank(), "C2", "index_vector_dim " + ivd + " out of range")) {
            return;
        }
        long indexVectorSize = ivd < s.rank() ? s.dim((int) ivd) : 1;
        check(
                dimEq(map.length, indexVectorSize),
                "C3",
                "size(start_index_map) must equal the index vector size " + indexVectorSize);
        check(unique(off) && sorted(off), "C4", "offset_dims must be sorted and unique");
        check(inRange(off, 0, r.rank()), "C5", "offset_dims out of range");
        check(
                unique(concat(col, ob)),
                "C6",
                "collapsed_slice_dims and operand_batching_dims must be unique");
        check(sorted(col), "C7", "collapsed_slice_dims must be sorted");
        boolean colInRange = check(inRange(col, 0, n), "C8", "collapsed_slice_dims out of range");
        check(sorted(ob), "C10", "operand_batching_dims must be sorted");
        boolean obInRange = check(inRange(ob, 0, n), "C11", "operand_batching_dims out of range");
        check(unique(sb), "C13", "start_indices_batching_dims must be unique");
        boolean sbInRange =
                check(inRange(sb, 0, s.rank()), "C14", "start_indices_batching_dims out of range");
        check(
                !contains(sb, ivd),
                "C15",
                "index_vector_dim must not be a start indices batching dimension");
        boolean sameBatch =
                check(
                        ob.length == sb.length,
                        "C16",
                        "operand_batching_dims and start_indices_batching_dims differ in size");
        if (sameBatch && obInRange && sbInRange) {
            for (int i = 0; i < ob.length; i++) {
                check(
                        dimEq(x.dim((int) ob[i]), s.dim((int) sb[i])),
                        "C17",
                        "batching dimension sizes differ");
            }
        }
        check(
                unique(concat(map, ob)),
                "C18",
                "start_index_map and operand_batching_dims must be unique");
        check(inRange(map, 0, n), "C19", "start_index_map out of range");
        if (sizes == null) {
            return;
        }
        if (!check(sizes.length == n, "C20", "size(slice_sizes) must equal rank(operand)")) {
            return;
        }
        if (colInRange) {
            for (long c : col) {
                check(
                        sizes[(int) c] <= 1,
                        "C9",
                        "slice_sizes of collapsed dimensions must be <= 1");
            }
        }
        if (obInRange) {
            for (long c : ob) {
                check(
                        sizes[(int) c] <= 1,
                        "C12",
                        "slice_sizes of batching dimensions must be <= 1");
            }
        }
        boolean sizesOk = true;
        for (int i = 0; i < n; i++) {
            sizesOk &= sizes[i] >= 0 && (x.dim(i) == DYN || sizes[i] <= x.dim(i));
        }
        if (!check(sizesOk, "C21", "need 0 <= slice_sizes <= shape(operand)")) {
            return;
        }
        // C22: result shape
        List<Long> batchSizes = new ArrayList<>();
        for (int i = 0; i < s.rank(); i++) {
            if (i != ivd) {
                batchSizes.add(s.dim(i));
            }
        }
        List<Long> offsetSizes = new ArrayList<>();
        for (int i = 0; i < n; i++) {
            if (!contains(col, i) && !contains(ob, i)) {
                offsetSizes.add(sizes[i]);
            }
        }
        if (!check(
                r.rank() == batchSizes.size() + offsetSizes.size(),
                "C22",
                "result rank must be " + (batchSizes.size() + offsetSizes.size()))) {
            return;
        }
        boolean ok = true;
        int bi = 0;
        int oi = 0;
        for (int d = 0; d < r.rank(); d++) {
            long want = contains(off, d) ? offsetSizes.get(oi++) : batchSizes.get(bi++);
            ok &= dimEq(r.dim(d), want);
        }
        check(ok, "C22", "result shape does not combine the batch and offset dimension sizes");
        if (!isQuantized(x)) {
            check(
                    elemEq(x.elementType(), r.elementType()),
                    "C23",
                    "result element type must match the operand");
        }
    }

    @Override
    public Void visitScatter(ScatterOp op) {
        ScatterDimensionNumbers dn = op.scatterDimensionNumbers();
        List<Value> inputs = op.inputs();
        List<Value> updates = op.updates();
        TensorType idx = tt(op.scatterIndices());
        if (!check(
                        !inputs.isEmpty() && inputs.size() == updates.size(),
                        "C5",
                        "needs N inputs and N updates")
                || idx == null) {
            return null;
        }
        TensorType in0 = tt(inputs.getFirst());
        TensorType up0 = tt(updates.getFirst());
        if (in0 == null || up0 == null) {
            return null;
        }
        for (Value v : inputs) {
            check(shapeEq(tt(v), in0), "C1", "inputs must have the same shape");
        }
        for (Value v : updates) {
            check(shapeEq(tt(v), up0), "C3", "updates must have the same shape");
        }
        long[] uw = dn.updateWindowDims();
        long[] iw = dn.insertedWindowDims();
        long[] ib = dn.inputBatchingDims();
        long[] sb = dn.scatterIndicesBatchingDims();
        long[] map = dn.scatterDimsToOperandDims();
        long ivd = dn.indexVectorDim();
        int n = in0.rank();
        check(
                n == uw.length + iw.length + ib.length,
                "C2",
                "rank(inputs[0]) must equal size(update_window_dims) + size(inserted_window_dims) +"
                        + " size(input_batching_dims)");
        for (int i = 0; i < inputs.size(); i++) {
            check(
                    elemEq(tt(updates.get(i)).elementType(), tt(inputs.get(i)).elementType()),
                    "C6",
                    "updates must have the input element types");
        }
        check(unique(uw) && sorted(uw), "C7", "update_window_dims must be sorted and unique");
        check(inRange(uw, 0, up0.rank()), "C8", "update_window_dims out of range");
        check(
                unique(concat(iw, ib)),
                "C9",
                "inserted_window_dims and input_batching_dims must be unique");
        check(sorted(iw), "C10", "inserted_window_dims must be sorted");
        check(inRange(iw, 0, n), "C11", "inserted_window_dims out of range");
        check(sorted(ib), "C12", "input_batching_dims must be sorted");
        boolean ibOk = check(inRange(ib, 0, n), "C13", "input_batching_dims out of range");
        check(unique(sb), "C14", "scatter_indices_batching_dims must be unique");
        boolean sbOk =
                check(
                        inRange(sb, 0, idx.rank()),
                        "C15",
                        "scatter_indices_batching_dims out of range");
        check(!contains(sb, ivd), "C16", "index_vector_dim must not be a batching dimension");
        if (check(ib.length == sb.length, "C17", "batching dimension counts differ")
                && ibOk
                && sbOk) {
            for (int i = 0; i < ib.length; i++) {
                check(
                        dimEq(in0.dim((int) ib[i]), idx.dim((int) sb[i])),
                        "C18",
                        "batching dimension sizes differ");
            }
        }
        if (check(ivd >= 0 && ivd <= idx.rank(), "C22", "index_vector_dim out of range")) {
            long vectorSize = ivd < idx.rank() ? idx.dim((int) ivd) : 1;
            check(
                    dimEq(map.length, vectorSize),
                    "C19",
                    "size(scatter_dims_to_operand_dims) must equal " + vectorSize);
        }
        check(
                unique(concat(map, ib)),
                "C20",
                "scatter_dims_to_operand_dims and input_batching_dims must be unique");
        check(inRange(map, 0, n), "C21", "scatter_dims_to_operand_dims out of range");
        // C4: update shape
        List<Long> scatterSizes = new ArrayList<>();
        for (int i = 0; i < idx.rank(); i++) {
            if (i != ivd) {
                scatterSizes.add(idx.dim(i));
            }
        }
        if (check(
                up0.rank() == scatterSizes.size() + uw.length,
                "C4",
                "updates rank must be " + (scatterSizes.size() + uw.length))) {
            List<Long> windowBounds = new ArrayList<>();
            for (int i = 0; i < n; i++) {
                if (!contains(iw, i) && !contains(ib, i)) {
                    windowBounds.add(in0.dim(i));
                }
            }
            int si = 0;
            int wi = 0;
            boolean ok = windowBounds.size() == uw.length;
            for (int d = 0; ok && d < up0.rank(); d++) {
                if (contains(uw, d)) {
                    long bound = windowBounds.get(wi++);
                    ok = up0.dim(d) == DYN || bound == DYN || up0.dim(d) <= bound;
                } else {
                    ok = dimEq(up0.dim(d), scatterSizes.get(si++));
                }
            }
            check(
                    ok,
                    "C4",
                    "updates shape does not combine the scatter and window dimension sizes");
        }
        List<ElementType> es =
                checkReducer(
                        op.updateComputation(),
                        elems(inputs),
                        elems(updates).stream().map(Verifier::scalar).toList(),
                        new long[0],
                        "C23");
        if (es != null) {
            for (int i = 0; i < op.operation().numResults(); i++) {
                TensorType r = tt(op.operation().result(i));
                check(r != null && shapeEq(r, in0), "C24", "results must have the input shape");
                check(
                        r != null && elemEq(r.elementType(), es.get(i)),
                        "C25",
                        "result element type must be E" + i);
            }
        }
        return null;
    }

    // ================================================================== reductions

    @Override
    public Void visitReduce(ReduceOp op) {
        List<Value> inputs = op.inputs();
        List<Value> inits = op.initValues();
        int n = inputs.size();
        if (!check(
                n > 0 && inits.size() == n && op.operation().numResults() == n,
                "C3",
                "needs N inputs, N init values and N results")) {
            return null;
        }
        TensorType in0 = tt(inputs.getFirst());
        if (in0 == null) {
            return null;
        }
        for (int i = 0; i < n; i++) {
            check(shapeEq(tt(inputs.get(i)), in0), "C1", "inputs must have the same shape");
            check(
                    elemEq(tt(inputs.get(i)).elementType(), tt(inits.get(i)).elementType()),
                    "C2",
                    "init value #" + i + " must have the input element type");
        }
        long[] dims = op.dimensions();
        if (!check(inRange(dims, 0, in0.rank()), "C4", "dimensions " + str(dims) + " out of range")
                || !check(unique(dims), "C5", "dimensions must be unique")) {
            return null;
        }
        List<ElementType> es =
                checkReducer(op.body(), elems(inputs), types(inits), resultDims(in0, dims), "C6");
        for (int i = 0; i < n; i++) {
            TensorType r = tt(op.operation().result(i));
            if (r == null) {
                continue;
            }
            boolean ok = r.rank() == in0.rank() - dims.length;
            for (int d = 0, k = 0; ok && d < in0.rank(); d++) {
                if (!contains(dims, d)) {
                    ok = dimEq(r.dim(k++), in0.dim(d));
                }
            }
            check(
                    ok,
                    "C7",
                    "result #" + i + " must have the input shape without the reduced dimensions");
            if (es != null) {
                check(
                        elemEq(r.elementType(), es.get(i)),
                        "C8",
                        "result #" + i + " element type must be E" + i);
            }
        }
        return null;
    }

    private static long[] resultDims(TensorType in, long[] reduced) {
        long[] out = new long[in.rank() - reduced.length];
        for (int d = 0, k = 0; d < in.rank() && k < out.length; d++) {
            if (!contains(reduced, d)) {
                out[k++] = in.dim(d);
            }
        }
        return out;
    }

    /** {@code num_windows} of reduce_window / select_and_scatter / convolution. */
    private static long windows(
            long input,
            long window,
            long stride,
            long baseDil,
            long winDil,
            long padLo,
            long padHi) {
        long dilatedInput = input == 0 ? 0 : (input - 1) * baseDil + 1;
        long padded = padLo + dilatedInput + padHi;
        long dilatedWindow = window == 0 ? 0 : (window - 1) * winDil + 1;
        if (padded <= 0 || dilatedWindow > padded) {
            return 0;
        }
        return (padded - dilatedWindow) / stride + 1;
    }

    private static long[][] padding(DenseElementsAttr d, int rank) {
        long[][] p = new long[rank][2];
        if (d != null) {
            for (int i = 0; i < rank && 2 * i + 1 < d.numElements(); i++) {
                p[i][0] = d.getLong(2 * i);
                p[i][1] = d.getLong(2 * i + 1);
            }
        }
        return p;
    }

    private static long[] orFilled(java.util.Optional<long[]> v, int n, long fill) {
        if (v.isPresent() && v.get().length > 0) {
            return v.get(); // the reference reads an empty array as the default
        }
        long[] out = new long[n];
        Arrays.fill(out, fill);
        return out;
    }

    @Override
    public Void visitReduceWindow(ReduceWindowOp op) {
        List<Value> inputs = op.inputs();
        int n = inputs.size();
        if (!check(
                n > 0 && op.initValues().size() == n && op.operation().numResults() == n,
                "C1",
                "needs N inputs, N init values and N results")) {
            return null;
        }
        TensorType in0 = tt(inputs.getFirst());
        if (in0 == null) {
            return null;
        }
        int rank = in0.rank();
        for (int i = 0; i < n; i++) {
            check(shapeEq(tt(inputs.get(i)), in0), "C2", "inputs must have the same shape");
            check(
                    elemEq(
                            tt(inputs.get(i)).elementType(),
                            tt(op.initValues().get(i)).elementType()),
                    "C3",
                    "init value #" + i + " must have the input element type");
        }
        long[] wd = op.windowDimensions();
        long[] ws = orFilled(op.windowStrides(), rank, 1);
        long[] bd = orFilled(op.baseDilations(), rank, 1);
        long[] wdl = orFilled(op.windowDilations(), rank, 1);
        boolean ok =
                check(wd.length == rank, "C4", "size(window_dimensions) must equal the input rank")
                        & check(
                                Arrays.stream(wd).allMatch(v -> v > 0),
                                "C5",
                                "window_dimensions must be positive")
                        & check(
                                ws.length == rank,
                                "C6",
                                "size(window_strides) must equal the input rank")
                        & check(
                                Arrays.stream(ws).allMatch(v -> v > 0),
                                "C7",
                                "window_strides must be positive")
                        & check(
                                bd.length == rank,
                                "C8",
                                "size(base_dilations) must equal the input rank")
                        & check(
                                Arrays.stream(bd).allMatch(v -> v > 0),
                                "C9",
                                "base_dilations must be positive")
                        & check(
                                wdl.length == rank,
                                "C10",
                                "size(window_dilations) must equal the input rank")
                        & check(
                                Arrays.stream(wdl).allMatch(v -> v > 0),
                                "C11",
                                "window_dilations must be positive");
        DenseElementsAttr pad = op.padding().orElse(null);
        if (pad != null) {
            ok &=
                    check(
                            pad.type().rank() == 2
                                    && pad.type().dim(0) == rank
                                    && pad.type().dim(1) == 2,
                            "C12",
                            "padding must have shape [" + rank + ", 2]");
        }
        List<ElementType> es =
                checkReducer(op.body(), elems(inputs), types(op.initValues()), wd, "C13");
        if (!ok) {
            return null;
        }
        long[][] p = padding(pad, rank);
        for (int i = 0; i < n; i++) {
            TensorType r = tt(op.operation().result(i));
            if (r == null) {
                continue;
            }
            boolean shape = r.rank() == rank;
            for (int d = 0; shape && d < rank; d++) {
                if (in0.dim(d) != DYN) {
                    shape =
                            dimEq(
                                    r.dim(d),
                                    windows(
                                            in0.dim(d),
                                            wd[d],
                                            ws[d],
                                            bd[d],
                                            wdl[d],
                                            p[d][0],
                                            p[d][1]));
                }
            }
            check(shape, "C15", "result #" + i + " shape must be the number of windows");
            if (es != null) {
                check(
                        elemEq(r.elementType(), es.get(i)),
                        "C16",
                        "result #" + i + " element type must be E" + i);
            }
        }
        return null;
    }

    @Override
    public Void visitSelectAndScatter(SelectAndScatterOp op) {
        TensorType x = tt(op.operand());
        TensorType src = tt(op.source());
        TensorType init = tt(op.initValue());
        TensorType r = tt(op.result());
        if (x == null || src == null || init == null || r == null) {
            return null;
        }
        int rank = x.rank();
        check(
                elemEq(x.elementType(), src.elementType()),
                "C1",
                "source must have the operand element type");
        check(
                elemEq(init.elementType(), x.elementType()),
                "C3",
                "init_value must have the operand element type");
        long[] wd = orFilled(op.windowDimensions(), rank, 1);
        long[] ws = orFilled(op.windowStrides(), rank, 1);
        boolean ok =
                check(
                                wd.length == rank,
                                "C4",
                                "size(window_dimensions) must equal the operand rank")
                        & check(
                                Arrays.stream(wd).allMatch(v -> v > 0),
                                "C5",
                                "window_dimensions must be positive")
                        & check(
                                ws.length == rank,
                                "C6",
                                "size(window_strides) must equal the operand rank")
                        & check(
                                Arrays.stream(ws).allMatch(v -> v > 0),
                                "C7",
                                "window_strides must be positive");
        DenseElementsAttr pad = op.padding().orElse(null);
        if (pad != null) {
            ok &=
                    check(
                            pad.type().rank() == 2
                                    && pad.type().dim(0) == rank
                                    && pad.type().dim(1) == 2,
                            "C8",
                            "padding must have shape [" + rank + ", 2]");
        }
        List<Type> selArgs = argumentTypes(op.select());
        List<Type> selRet = returned(op.select());
        check(
                selArgs != null
                        && selArgs.size() == 2
                        && selRet != null
                        && selRet.size() == 1
                        && tt(selRet.getFirst()) != null
                        && isBool(tt(selRet.getFirst()).elementType()),
                "C9",
                "select must have type (tensor<E>, tensor<E>) -> tensor<i1>");
        List<ElementType> es =
                checkReducer(
                        op.scatter(),
                        List.of(src.elementType()),
                        List.of(init),
                        new long[0],
                        "C10");
        if (ok && rank == src.rank()) {
            long[][] p = padding(pad, rank);
            boolean shape = true;
            for (int d = 0; shape && d < rank; d++) {
                if (x.dim(d) != DYN) {
                    shape =
                            dimEq(
                                    src.dim(d),
                                    windows(x.dim(d), wd[d], ws[d], 1, 1, p[d][0], p[d][1]));
                }
            }
            check(shape, "C2", "source shape must be the number of windows");
        }
        check(shapeEq(x, r), "C11", "result must have the operand shape");
        if (es != null) {
            check(elemEq(r.elementType(), es.getFirst()), "C12", "result element type must be E");
        }
        return null;
    }

    @Override
    public Void visitSort(SortOp op) {
        List<Value> in = op.inputs();
        if (!check(!in.isEmpty(), "C1", "needs at least one input")) {
            return null;
        }
        check(
                typesEq(op.operation().operandTypes(), op.operation().resultTypes()),
                "C2",
                "results must have the input types");
        TensorType first = tt(in.getFirst());
        if (first == null) {
            return null;
        }
        for (Value v : in) {
            check(shapeEq(tt(v), first), "C3", "inputs must have the same shape");
        }
        long dim = op.dimension().orElse(-1);
        check(
                dim >= -first.rank() && dim < first.rank(),
                "C4",
                "dimension " + dim + " out of range");
        List<Type> args = argumentTypes(op.comparator());
        List<Type> ret = returned(op.comparator());
        boolean ok =
                args != null
                        && args.size() == 2 * in.size()
                        && ret != null
                        && ret.size() == 1
                        && tt(ret.getFirst()) != null
                        && tt(ret.getFirst()).rank() == 0
                        && isBool(tt(ret.getFirst()).elementType());
        for (int i = 0; ok && i < in.size(); i++) {
            ElementType e = tt(in.get(i)).elementType();
            ok =
                    tt(args.get(2 * i)) != null
                            && elemEq(tt(args.get(2 * i)).elementType(), e)
                            && tt(args.get(2 * i + 1)) != null
                            && elemEq(tt(args.get(2 * i + 1)).elementType(), e);
        }
        check(ok, "C5", "comparator must have type (tensor<E0>, tensor<E0>, ...) -> tensor<i1>");
        return null;
    }

    @Override
    public Void visitMap(MapOp op) {
        List<Value> in = op.inputs();
        TensorType r = tt(op.result());
        if (!check(!in.isEmpty(), "C2", "needs at least one input") || r == null) {
            return null;
        }
        for (Value v : in) {
            check(shapeEq(tt(v), r), "C1", "inputs must have the result shape");
        }
        long[] dims = op.dimensions();
        long[] all = new long[tt(in.getFirst()).rank()];
        for (int i = 0; i < all.length; i++) {
            all[i] = i;
        }
        check(Arrays.equals(dims, all), "C3", "dimensions must be " + str(all));
        List<Type> args = argumentTypes(op.computation());
        List<Type> ret = returned(op.computation());
        boolean ok = args != null && args.size() == in.size() && ret != null && ret.size() == 1;
        for (int i = 0; ok && i < in.size(); i++) {
            ok =
                    tt(args.get(i)) != null
                            && tt(args.get(i)).rank() == 0
                            && elemEq(tt(args.get(i)).elementType(), tt(in.get(i)).elementType());
        }
        ok =
                ok
                        && tt(ret.getFirst()) != null
                        && tt(ret.getFirst()).rank() == 0
                        && elemEq(tt(ret.getFirst()).elementType(), r.elementType());
        check(ok, "C4", "computation must have type (tensor<E0>, ...) -> tensor<E'>");
        return null;
    }

    // ================================================================== contractions

    @Override
    public Void visitDotGeneral(DotGeneralOp op) {
        TensorType l = tt(op.lhs());
        TensorType rh = tt(op.rhs());
        TensorType r = tt(op.result());
        if (l == null || rh == null || r == null) {
            return null;
        }
        DotDimensionNumbers dn = op.dotDimensionNumbers();
        long[] lb = dn.lhsBatchingDimensions();
        long[] rb = dn.rhsBatchingDimensions();
        long[] lc = dn.lhsContractingDimensions();
        long[] rc = dn.rhsContractingDimensions();
        boolean ok =
                check(lb.length == rb.length, "C1", "batching dimension counts differ")
                        & check(lc.length == rc.length, "C2", "contracting dimension counts differ")
                        & check(
                                unique(concat(lb, lc)),
                                "C3",
                                "lhs batching and contracting dimensions must be unique")
                        & check(
                                unique(concat(rb, rc)),
                                "C4",
                                "rhs batching and contracting dimensions must be unique")
                        & check(
                                inRange(lb, 0, l.rank()),
                                "C5",
                                "lhs_batching_dimensions out of range")
                        & check(
                                inRange(lc, 0, l.rank()),
                                "C6",
                                "lhs_contracting_dimensions out of range")
                        & check(
                                inRange(rb, 0, rh.rank()),
                                "C7",
                                "rhs_batching_dimensions out of range")
                        & check(
                                inRange(rc, 0, rh.rank()),
                                "C8",
                                "rhs_contracting_dimensions out of range");
        if (!ok) {
            return null;
        }
        for (int i = 0; i < lb.length; i++) {
            check(
                    dimEq(l.dim((int) lb[i]), rh.dim((int) rb[i])),
                    "C9",
                    "batching dimension sizes differ");
        }
        boolean extended = op.operation().numOperands() > 2; // sparsity metadata / block scales
        for (int i = 0; i < lc.length && !extended; i++) {
            check(
                    dimEq(l.dim((int) lc[i]), rh.dim((int) rc[i])),
                    "C10",
                    "contracting dimension sizes differ");
        }
        op.precisionConfig()
                .ifPresent(
                        p -> {
                            if (p instanceof ArrayAttr a) {
                                check(
                                        a.elements().size() <= 2,
                                        "C11",
                                        "precision_config must have at most 2 elements");
                            }
                        });
        List<Long> shape = new ArrayList<>();
        for (long d : lb) {
            shape.add(l.dim((int) d));
        }
        for (int d = 0; d < l.rank(); d++) {
            if (!contains(lb, d) && !contains(lc, d)) {
                shape.add(l.dim(d));
            }
        }
        for (int d = 0; d < rh.rank(); d++) {
            if (!contains(rb, d) && !contains(rc, d)) {
                shape.add(rh.dim(d));
            }
        }
        boolean shapeOk = r.rank() == shape.size();
        for (int i = 0; shapeOk && i < shape.size(); i++) {
            shapeOk = dimEq(r.dim(i), shape.get(i));
        }
        check(shapeOk, "C12", "result shape must be " + shape);
        checkQuantizedContraction(l, rh, r, rc, -1, -1);
        if (op.algorithm().orElse(null) instanceof com.qxotic.jota.stablehlo.attr.DialectAttr alg
                && alg.hasParams()) {
            for (String[] p :
                    new String[][] {
                        {"lhs_component_count", "C22"},
                        {"rhs_component_count", "C23"},
                        {"num_primitive_operations", "C24"}
                    }) {
                long v =
                        alg.param(p[0])
                                .map(Verifier::longs)
                                .map(x -> x.length == 0 ? 1 : x[0])
                                .orElse(1L);
                check(v > 0, p[1], p[0].replace('_', ' ') + " must be positive");
            }
            op.precisionConfig()
                    .ifPresent(
                            pc -> {
                                if (pc instanceof ArrayAttr a) {
                                    check(
                                            a.elements().stream()
                                                    .allMatch(e -> enumName(e).equals("DEFAULT")),
                                            "C21",
                                            "must specify DEFAULT precision config when algorithm"
                                                    + " is set");
                                }
                            });
        }
        if (!isQuantized(l) && !isQuantized(rh)) {
            check(
                    elemEq(l.elementType(), rh.elementType())
                            || isFp8(l.elementType()) && isFp8(rh.elementType()),
                    "C13",
                    "lhs and rhs must have the same element type");
        }
        return null;
    }

    private static boolean isFp8(ElementType e) {
        return e instanceof FloatType f && f.format().bitWidth() <= 8;
    }

    @Override
    public Void visitDot(DotOp op) {
        TensorType l = tt(op.lhs());
        TensorType rh = tt(op.rhs());
        if (l == null || rh == null) {
            return null;
        }
        if (check(
                l.rank() >= 1 && l.rank() <= 2 && rh.rank() >= 1 && rh.rank() <= 2,
                "dot",
                "lhs and rhs must have rank 1 or 2")) {
            check(
                    dimEq(l.dim(l.rank() - 1), rh.dim(0)),
                    "dot",
                    "contracting dimension sizes differ");
            TensorType r = tt(op.operation().result(0));
            if (r != null) {
                List<Long> want = new ArrayList<>();
                for (int i = 0; i < l.rank() - 1; i++) {
                    want.add(l.dim(i));
                }
                for (int i = 1; i < rh.rank(); i++) {
                    want.add(rh.dim(i));
                }
                boolean ok = r.rank() == want.size();
                for (int i = 0; ok && i < want.size(); i++) {
                    ok = dimEq(r.dim(i), want.get(i));
                }
                check(
                        ok,
                        "dot",
                        "inferred shape "
                                + want
                                + " is incompatible with the result type "
                                + r.mlir());
            }
        }
        return null;
    }

    @Override
    public Void visitConvolution(ConvolutionOp op) {
        checkConvolution(
                op.operation(),
                tt(op.lhs()),
                tt(op.rhs()),
                tt(op.result()),
                op.dimensionNumbers(),
                op.windowStrides(),
                op.padding().orElse(null),
                op.lhsDilation(),
                op.rhsDilation(),
                op.windowReversal(),
                op.featureGroupCount(),
                op.batchGroupCount(),
                op.precisionConfig());
        return null;
    }

    @Override
    public Void visitDynamicConv(DynamicConvOp op) {
        TensorType pad = tt(op.padding());
        TensorType lhs = tt(op.lhs());
        if (pad != null && lhs != null && lhs.rank() >= 2) {
            check(
                    pad.rank() == 2 && pad.dim(0) == lhs.rank() - 2 && pad.dim(1) == 2,
                    "C4",
                    "padding must have shape [" + (lhs.rank() - 2) + ", 2]");
        }
        checkConvolution(
                op.operation(),
                tt(op.lhs()),
                tt(op.rhs()),
                null,
                op.dimensionNumbers(),
                op.windowStrides(),
                null,
                op.lhsDilation(),
                op.rhsDilation(),
                op.windowReversal(),
                op.featureGroupCount(),
                op.batchGroupCount(),
                op.precisionConfig());
        return null;
    }

    private void checkConvolution(
            Operation operation,
            TensorType l,
            TensorType rh,
            TensorType r,
            ConvDimensionNumbers dn,
            java.util.Optional<long[]> strides,
            DenseElementsAttr pad,
            java.util.Optional<long[]> lhsDil,
            java.util.Optional<long[]> rhsDil,
            java.util.Optional<Attribute> reversal,
            long fgc,
            long bgc,
            java.util.Optional<Attribute> precision) {
        if (l == null || rh == null) {
            return;
        }
        int n = l.rank();
        if (!check(n == rh.rank(), "C1", "lhs and rhs must have the same rank")
                || !check(n >= 2, "C1", "rank must be >= 2")) {
            return;
        }
        int s = n - 2;
        long[] ws = orFilled(strides, s, 1);
        long[] ld = orFilled(lhsDil, s, 1);
        long[] rd = orFilled(rhsDil, s, 1);
        boolean ok =
                check(ws.length == s, "C2", "size(window_strides) must be " + s)
                        & check(
                                Arrays.stream(ws).allMatch(v -> v > 0),
                                "C3",
                                "window_strides must be positive")
                        & check(ld.length == s, "C5", "size(lhs_dilation) must be " + s)
                        & check(
                                Arrays.stream(ld).allMatch(v -> v > 0),
                                "C6",
                                "lhs_dilation must be positive")
                        & check(rd.length == s, "C7", "size(rhs_dilation) must be " + s)
                        & check(
                                Arrays.stream(rd).allMatch(v -> v > 0),
                                "C8",
                                "rhs_dilation must be positive");
        if (pad != null && pad.numElements() == 0) {
            pad = null;
        }
        if (pad != null) {
            ok &=
                    check(
                            pad.type().rank() == 2
                                    && pad.type().dim(0) == s
                                    && pad.type().dim(1) == 2,
                            "C4",
                            "padding must have shape [" + s + ", 2]");
        }
        if (reversal.isPresent()
                && reversal.get() instanceof com.qxotic.jota.stablehlo.attr.DenseArrayAttr rv) {
            ok &=
                    check(
                            rv.size() == s || rv.size() == 0,
                            "C9",
                            "size(window_reversal) must be " + s);
        }
        ok &=
                check(fgc > 0, "C21", "feature_group_count must be positive")
                        & check(bgc > 0, "C22", "batch_group_count must be positive")
                        & check(
                                fgc == 1 || bgc == 1,
                                "C23",
                                "feature_group_count or batch_group_count must be 1");
        long[] inDims =
                concat(
                        concat(new long[] {dn.inputBatchDimension()}, dn.inputSpatialDimensions()),
                        new long[] {dn.inputFeatureDimension()});
        long[] kDims =
                concat(
                        dn.kernelSpatialDimensions(),
                        new long[] {
                            dn.kernelInputFeatureDimension(), dn.kernelOutputFeatureDimension()
                        });
        long[] outDims =
                concat(
                        concat(
                                new long[] {dn.outputBatchDimension()},
                                dn.outputSpatialDimensions()),
                        new long[] {dn.outputFeatureDimension()});
        boolean dimsOk =
                check(
                                dn.inputSpatialDimensions().length == s,
                                "C12",
                                "size(input_spatial_dimensions) must be " + s)
                        & check(
                                unique(inDims) && inRange(inDims, 0, n),
                                "C13",
                                "input dimensions must be unique and in range")
                        & check(
                                dn.kernelSpatialDimensions().length == s,
                                "C17",
                                "size(kernel_spatial_dimensions) must be " + s)
                        & check(
                                unique(kDims) && inRange(kDims, 0, n),
                                "C18",
                                "kernel dimensions must be unique and in range")
                        & check(
                                dn.outputSpatialDimensions().length == s,
                                "C19",
                                "size(output_spatial_dimensions) must be " + s)
                        & check(
                                unique(outDims) && inRange(outDims, 0, n),
                                "C20",
                                "output dimensions must be unique and in range");
        if (precision.isPresent() && precision.get() instanceof ArrayAttr a) {
            check(a.elements().size() <= 2, "C24", "precision_config must have at most 2 elements");
        }
        if (!isQuantized(l) && !isQuantized(rh)) {
            check(
                    elemEq(l.elementType(), rh.elementType())
                            || isFp8(l.elementType()) && isFp8(rh.elementType()),
                    "C27",
                    "expects lhs and rhs to have compatible element types");
        } else if (r != null) {
            checkQuantizedContraction(
                    l,
                    rh,
                    r,
                    new long[0],
                    dn.kernelOutputFeatureDimension(),
                    dn.outputFeatureDimension());
        }
        if (!ok || !dimsOk || fgc <= 0 || bgc <= 0) {
            return;
        }
        long batch = l.dim((int) dn.inputBatchDimension());
        long feature = l.dim((int) dn.inputFeatureDimension());
        long kIn = rh.dim((int) dn.kernelInputFeatureDimension());
        long kOut = rh.dim((int) dn.kernelOutputFeatureDimension());
        if (batch != DYN) {
            check(
                    batch % bgc == 0,
                    "C10",
                    "input batch dimension must be divisible by batch_group_count");
        }
        if (feature != DYN) {
            check(
                    feature % fgc == 0,
                    "C11",
                    "input feature dimension must be divisible by feature_group_count");
            if (kIn != DYN) {
                check(
                        kIn == feature / fgc,
                        "C14",
                        "kernel input feature dimension must be " + feature / fgc);
            }
        }
        if (kOut != DYN) {
            check(
                    kOut % bgc == 0,
                    "C15",
                    "kernel output feature dimension must be divisible by batch_group_count");
            check(
                    kOut % fgc == 0,
                    "C16",
                    "kernel output feature dimension must be divisible by feature_group_count");
        }
        if (r == null) {
            return;
        }
        if (!check(r.rank() == n, "C26", "result rank must be " + n)) {
            return;
        }
        long[][] p = padding(pad, s);
        boolean shape =
                dimEq(r.dim((int) dn.outputBatchDimension()), batch == DYN ? DYN : batch / bgc)
                        && dimEq(r.dim((int) dn.outputFeatureDimension()), kOut);
        for (int i = 0; shape && i < s; i++) {
            long in = l.dim((int) dn.inputSpatialDimensions()[i]);
            long k = rh.dim((int) dn.kernelSpatialDimensions()[i]);
            if (in != DYN && k != DYN) {
                shape =
                        dimEq(
                                r.dim((int) dn.outputSpatialDimensions()[i]),
                                windows(in, k, ws[i], ld[i], rd[i], p[i][0], p[i][1]));
            }
        }
        check(shape, "C25", "result shape does not match the convolution");
    }

    @Override
    public Void visitFft(FftOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        long[] len = op.fftLength();
        check(len.length <= x.rank(), "C1", "size(fft_length) must be <= rank(operand)");
        check(len.length >= 1 && len.length <= 3, "C3", "size(fft_length) must be between 1 and 3");
        FftType type = op.fftType();
        ElementType xe = x.elementType();
        ElementType re = r.elementType();
        boolean types =
                switch (type) {
                    case FFT, IFFT -> xe instanceof ComplexType && xe.equals(re);
                    case RFFT ->
                            xe instanceof FloatType f
                                    && re instanceof ComplexType c
                                    && c.elementType().equals(f)
                                    && (f.format() == FloatFormat.F16
                                            || f.format() == FloatFormat.BF16
                                            || f.format() == FloatFormat.F32
                                            || f.format() == FloatFormat.F64);
                    case IRFFT ->
                            xe instanceof ComplexType c
                                    && re instanceof FloatType f
                                    && c.elementType().equals(f);
                };
        if (!check(
                types,
                "C2",
                "element types " + xe.mlir() + " -> " + re.mlir() + " are invalid for " + type)) {
            return null;
        }
        if (len.length > x.rank() || r.rank() != x.rank()) {
            check(r.rank() == x.rank(), "C5", "result must have the operand rank");
            return null;
        }
        TensorType real = type == FftType.RFFT ? x : type == FftType.IRFFT ? r : null;
        if (real != null) {
            boolean ok = true;
            for (int i = 0; i < len.length; i++) {
                ok &= dimEq(real.dim(real.rank() - len.length + i), len[i]);
            }
            check(
                    ok,
                    "C4",
                    "the trailing dimensions of the real tensor must equal fft_length " + str(len));
        }
        boolean shape = true;
        int last = x.rank() - 1;
        for (int i = 0; shape && i < x.rank(); i++) {
            if (i == last && type == FftType.RFFT && x.dim(i) != DYN) {
                shape = dimEq(r.dim(i), x.dim(i) == 0 ? 0 : x.dim(i) / 2 + 1);
            } else if (i == last && type == FftType.IRFFT && r.dim(i) != DYN) {
                shape = dimEq(x.dim(i), r.dim(i) == 0 ? 0 : r.dim(i) / 2 + 1);
            } else {
                shape = dimEq(r.dim(i), x.dim(i));
            }
        }
        check(shape, "C5", "result shape does not match the fft");
        return null;
    }

    // ================================================================== linear algebra & norms

    @Override
    public Void visitCholesky(CholeskyOp op) {
        TensorType a = tt(op.a());
        TensorType r = tt(op.result());
        if (a == null || r == null) {
            return null;
        }
        check(typeEq(a, r), "C1", "result must have the operand type");
        if (check(a.rank() >= 2, "C2", "rank must be >= 2")) {
            check(
                    dimEq(a.dim(a.rank() - 2), a.dim(a.rank() - 1)),
                    "C3",
                    "the last two dimensions must be equal");
        }
        return null;
    }

    @Override
    public Void visitTriangularSolve(TriangularSolveOp op) {
        TensorType a = tt(op.a());
        TensorType b = tt(op.b());
        TensorType r = tt(op.result());
        if (a == null || b == null || r == null) {
            return null;
        }
        check(
                elemEq(a.elementType(), b.elementType()),
                "C1",
                "a and b must have the same element type");
        if (!check(
                a.rank() >= 2 && a.rank() == b.rank(),
                "C2",
                "a and b must have the same rank >= 2")) {
            return null;
        }
        int rank = a.rank();
        boolean ok = true;
        for (int i = 0; i < rank - 2; i++) {
            ok &= dimEq(a.dim(i), b.dim(i));
        }
        ok &=
                dimEq(a.dim(rank - 2), a.dim(rank - 1))
                        && dimEq(a.dim(rank - 1), b.dim(op.leftSide() ? rank - 2 : rank - 1));
        check(ok, "C3", "shapes of a and b are incompatible");
        check(typeEq(b, r), "C4", "result must have the type of b");
        check(
                op.transposeA() != com.qxotic.jota.stablehlo.attr.Transpose.TRANSPOSE_INVALID,
                "type",
                "invalid transpose option value for triangular solve");
        return null;
    }

    private void checkFeature(
            TensorType x,
            long featureIndex,
            List<TensorType> perFeature,
            String rangeC,
            String sizeC) {
        if (!check(
                featureIndex >= 0 && featureIndex < x.rank(),
                rangeC,
                "feature_index out of range")) {
            return;
        }
        long f = x.dim((int) featureIndex);
        for (TensorType t : perFeature) {
            if (t != null
                    && !check(
                            t.rank() == 1 && dimEq(t.dim(0), f),
                            sizeC,
                            "per-feature operands must be 1-D of size " + f)) {
                return;
            }
        }
    }

    @Override
    public Void visitBatchNormInference(BatchNormInferenceOp op) {
        TensorType x = tt(op.operand());
        if (x == null) {
            return null;
        }
        checkFeature(
                x,
                op.featureIndex(),
                List.of(tt(op.scale()), tt(op.offset()), tt(op.mean()), tt(op.variance())),
                "C1",
                "C3");
        check(typeEq(x, op.result().type()), "C7", "result must have the operand type");
        return null;
    }

    @Override
    public Void visitBatchNormTraining(BatchNormTrainingOp op) {
        TensorType x = tt(op.operand());
        if (x == null) {
            return null;
        }
        checkFeature(x, op.featureIndex(), List.of(tt(op.scale()), tt(op.offset())), "C1", "C3");
        Operation o = op.operation();
        if (o.numResults() == 3) {
            check(typeEq(o.result(0).type(), x), "C7", "output must have the operand type");
            checkFeature(
                    x, op.featureIndex(), List.of(tt(o.result(1)), tt(o.result(2))), "C1", "C5");
        }
        return null;
    }

    @Override
    public Void visitBatchNormGrad(BatchNormGradOp op) {
        TensorType x = tt(op.operand());
        if (x == null) {
            return null;
        }
        checkFeature(
                x,
                op.featureIndex(),
                List.of(tt(op.scale()), tt(op.mean()), tt(op.variance())),
                "C1",
                "C5");
        check(shapeEq(tt(op.gradOutput()), x), "C3", "grad_output must have the operand shape");
        Operation o = op.operation();
        if (o.numResults() == 3) {
            check(shapeEq(tt(o.result(0)), x), "C3", "grad_operand must have the operand shape");
            check(
                    shapeEq(tt(o.result(1)), tt(op.scale()))
                            && shapeEq(tt(o.result(2)), tt(op.scale())),
                    "C4",
                    "grad_scale and grad_offset must have the scale shape");
        }
        return null;
    }

    @Override
    public Void visitRng(RngOp op) {
        TensorType a = tt(op.a());
        TensorType b = tt(op.b());
        TensorType r = tt(op.result());
        if (a == null || b == null || r == null) {
            return null;
        }
        check(
                elemEq(a.elementType(), b.elementType())
                        && elemEq(a.elementType(), r.elementType()),
                "C1",
                "a, b and result must have the same element type");
        check(a.rank() == 0, "type", "operand #0 must be a 0D tensor");
        check(b.rank() == 0, "type", "operand #1 must be a 0D tensor");
        TensorType shape = tt(op.shape());
        if (check(
                shape != null && shape.rank() == 1 && shape.hasStaticShape(),
                "type",
                "operand #2 must be a statically shaped 1-dimensional tensor")) {
            check(dimEq(shape.dim(0), r.rank()), "C3", "result rank must equal size(shape)");
            checkShapeOperand(op.shape(), r, "C3", true);
        }
        if (op.rngDistribution() == com.qxotic.jota.stablehlo.attr.RngDistribution.NORMAL) {
            check(
                    a.elementType() instanceof FloatType,
                    "C2",
                    "NORMAL distribution requires floats");
        }
        return null;
    }

    @Override
    public Void visitRngBitGenerator(RngBitGeneratorOp op) {
        Operation o = op.operation();
        if (o.numResults() >= 1) {
            check(
                    typeEq(op.initialState().type(), o.result(0).type()),
                    "C1",
                    "output_state must have the initial_state type");
        }
        // the state size is implementation-defined; the reference does not check C2
        return null;
    }

    // ================================================================== control flow

    @Override
    public Void visitIf(IfOp op) {
        List<Type> t = returned(op.trueBranch());
        List<Type> f = returned(op.falseBranch());
        check(
                argumentTypes(op.trueBranch()) != null
                        && argumentTypes(op.trueBranch()).isEmpty()
                        && argumentTypes(op.falseBranch()) != null
                        && argumentTypes(op.falseBranch()).isEmpty(),
                "C1",
                "branches must not take arguments");
        if (check(typesEq(t, f), "C2", "branches must return the same types")) {
            check(
                    typesEq(op.operation().resultTypes(), t),
                    "C3",
                    "results must have the branch output types");
        }
        TensorType p = tt(op.pred());
        if (p != null) {
            check(
                    p.rank() == 0 && isBool(p.elementType()),
                    "if",
                    "pred must be a rank-0 i1 tensor");
        }
        return null;
    }

    @Override
    public Void visitCase(CaseOp op) {
        List<Region> branches = op.branches();
        if (!check(!branches.isEmpty(), "C1", "needs at least one branch")) {
            return null;
        }
        List<Type> first = returned(branches.getFirst());
        for (Region b : branches) {
            check(
                    argumentTypes(b) != null && argumentTypes(b).isEmpty(),
                    "C2",
                    "branches must not take arguments");
            if (!check(typesEq(returned(b), first), "C3", "branches must return the same types")) {
                return null;
            }
        }
        check(
                typesEq(op.operation().resultTypes(), first),
                "C4",
                "results must have the branch output types");
        TensorType idx = tt(op.index());
        if (idx != null) {
            check(
                    idx.rank() == 0
                            && idx.elementType() instanceof IntegerType it
                            && it.bitWidth() == 32
                            && !it.isUnsigned(),
                    "case",
                    "index must be a rank-0 si32 tensor");
        }
        return null;
    }

    @Override
    public Void visitWhile(WhileOp op) {
        List<Type> in = op.operation().operandTypes();
        List<Type> condRet = returned(op.cond());
        check(
                typesEq(argumentTypes(op.cond()), in)
                        && condRet != null
                        && condRet.size() == 1
                        && tt(condRet.getFirst()) != null
                        && tt(condRet.getFirst()).rank() == 0
                        && isBool(tt(condRet.getFirst()).elementType()),
                "C1",
                "cond must have type (T0, ...) -> tensor<i1>");
        check(
                typesEq(argumentTypes(op.body()), in) && typesEq(returned(op.body()), in),
                "C2",
                "body must have type (T0, ...) -> (T0, ...)");
        check(
                typesEq(op.operation().resultTypes(), in),
                "C3",
                "results must have the operand types");
        return null;
    }

    // ================================================================== collectives

    private static long[][] groups(DenseElementsAttr d) {
        if (d == null || d.type().rank() != 2) {
            return null;
        }
        int rows = (int) d.type().dim(0);
        int cols = (int) d.type().dim(1);
        long[][] g = new long[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                g[i][j] = d.getLong((long) i * cols + j);
            }
        }
        return g;
    }

    /**
     * {@code replica_groups} as the reference checks them ({@code verifyReplicaGroups}): a rank-2
     * tensor of unique ids covering {@code 0..n-1}, {@code -1} padding only when groups may differ
     * in size, a given group size, and a positive (or non-negative) channel id when {@code
     * use_global_device_ids} is set. Mesh-axes groups are not checked.
     */
    private void checkGroups(String c, boolean sameSize, long groupSize, boolean channelMayBeZero) {
        checkGroups(c, sameSize, groupSize, channelMayBeZero, true);
    }

    /** {@code dense}: ids must cover {@code 0..n-1} (all but collective_broadcast). */
    private void checkGroups(
            String c, boolean sameSize, long groupSize, boolean channelMayBeZero, boolean dense) {
        Attribute raw = current.attribute("replica_groups").orElse(null);
        boolean global =
                current.attribute("use_global_device_ids").isPresent()
                        && !(current.attribute("use_global_device_ids").get()
                                        instanceof com.qxotic.jota.stablehlo.attr.BoolAttr b
                                && !b.value());
        if (raw instanceof DenseElementsAttr d) {
            if (!check(d.type().rank() == 2, c, "replica groups should be a rank 2 tensor")) {
                return;
            }
            if (global
                    && !check(
                            d.numElements() > 0,
                            c,
                            "replica groups cannot be empty with use_global_device_ids")) {
                return;
            }
            Set<Long> seen = new HashSet<>();
            for (long i = 0; i < d.numElements(); i++) {
                long id = d.getLong(i);
                if (id == -1) {
                    if (!check(!sameSize, c, "Invalid replica id -1")) {
                        return;
                    }
                    continue;
                }
                if (!check(seen.add(id), c, "replica id #" + id + " seen more than once")) {
                    return;
                }
            }
            for (long id = 0; dense && id < seen.size(); id++) {
                if (!check(
                        seen.contains(id),
                        c,
                        "replica id #" + id + " not seen in replica groups")) {
                    return;
                }
            }
            if (sameSize && groupSize > 0) {
                long rows = d.type().dim(0);
                check(
                        rows != 0 && d.numElements() / rows == groupSize,
                        c,
                        "group size of replica_groups must be " + groupSize);
            }
        }
        if (global) {
            long channel = channelHandle("handle");
            check(
                    channelMayBeZero ? channel >= 0 : channel > 0,
                    "channel",
                    "channel_id must be "
                            + (channelMayBeZero ? "non-negative" : "positive")
                            + " when use_global_device_ids is set");
        }
    }

    /** A parameter of {@code channel_handle}, 0 when absent. */
    private long channelHandle(String param) {
        return current.attribute("channel_handle")
                .filter(
                        a ->
                                a instanceof com.qxotic.jota.stablehlo.attr.DialectAttr d
                                        && d.hasParams())
                .flatMap(a -> ((com.qxotic.jota.stablehlo.attr.DialectAttr) a).param(param))
                .map(
                        a ->
                                a instanceof com.qxotic.jota.stablehlo.attr.IntegerAttr i
                                        ? i.value()
                                        : 0L)
                .orElse(0L);
    }

    @Override
    public Void visitAllGather(AllGatherOp op) {
        checkGroups("C2", true, -1, true);
        long dim = op.allGatherDim();
        Operation o = op.operation();
        check(o.numOperands() == o.numResults(), "C6", "needs one result per operand");
        for (int i = 0; i < o.numOperands() && i < o.numResults(); i++) {
            TensorType x = tt(o.operand(i));
            TensorType r = tt(o.result(i));
            if (x == null
                    || r == null
                    || !check(dim >= 0 && dim < x.rank(), "C1", "all_gather_dim out of range")) {
                continue;
            }
            // as the reference: the gathered size is a multiple of the operand size (group size
            // depends on the replica configuration), other dimensions are equal
            boolean ok = r.rank() == x.rank() && elemEq(r.elementType(), x.elementType());
            for (int d = 0; ok && d < x.rank(); d++) {
                long a = x.dim(d);
                long b = r.dim(d);
                ok = d != dim ? dimEq(a, b) : a == DYN || b == DYN || a != 0 && b % a == 0;
            }
            check(ok, "C6", "result type does not match the gathered operand");
        }
        return null;
    }

    @Override
    public Void visitAllReduce(AllReduceOp op) {
        checkGroups("C1", false, -1, false);
        Operation o = op.operation();
        List<ElementType> es =
                checkReducer(
                        op.computation(),
                        elems(List.of(o.operand(0))),
                        List.of(scalar(tt(o.operand(0)).elementType())),
                        new long[0],
                        "C5");
        for (int i = 0; i < o.numOperands() && i < o.numResults(); i++) {
            check(
                    shapeEq(tt(o.operand(i)), tt(o.result(i))),
                    "C6",
                    "results must have the operand shapes");
            if (es != null) {
                check(
                        elemEq(tt(o.result(i)).elementType(), es.getFirst()),
                        "C7",
                        "result element type must be E");
            }
        }
        return null;
    }

    @Override
    public Void visitReduceScatter(ReduceScatterOp op) {
        checkGroups("C3", true, -1, false);
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        long dim = op.scatterDimension();
        if (!check(dim >= 0 && dim < x.rank(), "C2", "scatter_dimension out of range")) {
            return null;
        }
        List<ElementType> es =
                checkReducer(
                        op.computation(),
                        List.of(x.elementType()),
                        List.of(scalar(x.elementType())),
                        new long[0],
                        "C7");
        boolean ok = r.rank() == x.rank();
        for (int d = 0; ok && d < x.rank(); d++) {
            long a = x.dim(d);
            long b = r.dim(d);
            ok = d != dim ? dimEq(a, b) : a == DYN || b == DYN || a != 0 && b != 0 && a % b == 0;
        }
        check(ok, "C8", "result shape does not match the scattered operand");
        if (es != null) {
            check(elemEq(r.elementType(), es.getFirst()), "C9", "result element type must be E");
        }
        return null;
    }

    @Override
    public Void visitAllToAll(AllToAllOp op) {
        checkGroups("C5", true, op.splitCount(), true);
        long groupSize = -1;
        long split = op.splitDimension();
        long concatDim = op.concatDimension();
        long count = op.splitCount();
        check(count > 0, "C4", "split_count must be positive");
        if (groupSize > 0) {
            check(groupSize == count, "C8", "replica group size must equal split_count");
        }
        Operation o = op.operation();
        check(
                o.numOperands() == o.numResults(),
                "C9",
                "requires the same number of operands and results");
        for (int i = 0; i < o.numOperands(); i++) {
            TensorType x = tt(o.operand(i));
            if (x == null) {
                continue;
            }
            if (i < o.numResults() && tt(o.result(i)) != null) {
                check(
                        elemEq(x.elementType(), tt(o.result(i)).elementType()),
                        "C9",
                        "requires the same element type for operand and result at index " + i);
            }
            boolean ok =
                    check(split >= 0 && split < x.rank(), "C1", "split_dimension out of range")
                            & check(
                                    concatDim >= 0 && concatDim < x.rank(),
                                    "C3",
                                    "concat_dimension out of range");
            if (!ok || count <= 0) {
                continue;
            }
            if (x.dim((int) split) != DYN) {
                check(
                        x.dim((int) split) % count == 0,
                        "C2",
                        "split dimension must be divisible by split_count");
            }
            if (i < o.numResults() && tt(o.result(i)) instanceof TensorType r) {
                boolean shape = r.rank() == x.rank();
                for (int d = 0; shape && d < x.rank(); d++) {
                    long want = x.dim(d);
                    if (want != DYN && split != concatDim) {
                        if (d == split) {
                            want = want / count;
                        } else if (d == concatDim) {
                            want = want * count;
                        }
                    }
                    shape = dimEq(r.dim(d), want);
                }
                check(shape, "C9", "result shape does not match the all_to_all");
            }
        }
        return null;
    }

    @Override
    public Void visitCollectivePermute(CollectivePermuteOp op) {
        DenseElementsAttr pairs = op.sourceTargetPairs();
        if (check(
                pairs.type().rank() == 2 && pairs.type().dim(1) == 2,
                "C1",
                "source_target_pairs must have shape [N, 2]")) {
            long[][] g = groups(pairs);
            Set<Long> sources = new HashSet<>();
            Set<Long> targets = new HashSet<>();
            for (long[] p : g) {
                check(
                        p[0] >= 0 && p[1] >= 0,
                        "C4",
                        "replica ids in source_target_pairs must be >= 0");
                check(sources.add(p[0]), "C2", "source " + p[0] + " appears more than once");
                check(targets.add(p[1]), "C3", "target " + p[1] + " appears more than once");
            }
        }
        check(
                typeEq(op.operand().type(), op.result().type()),
                "C5",
                "result must have the operand type");
        return null;
    }

    @Override
    public Void visitCollectiveBroadcast(CollectiveBroadcastOp op) {
        checkGroups("C1", false, -1, true, false);
        Operation o = op.operation();
        boolean dynamicRoot = o.attribute("has_dynamic_root").isPresent();
        int data = o.numOperands() - (dynamicRoot ? 1 : 0);
        check(data >= 1, dynamicRoot ? "C4" : "C3", "needs at least one data operand");
        if (dynamicRoot && o.numOperands() >= 1) {
            TensorType root = tt(o.operand(o.numOperands() - 1));
            if (check(
                    root != null
                            && root.rank() == 1
                            && root.elementType() instanceof IntegerType it
                            && it.bitWidth() == 32,
                    "C4",
                    "the root operand must be a rank-1 i32 tensor")) {
                check(
                        dimEq(root.dim(0), data),
                        "C4",
                        "the root operand must have one element per data operand");
            }
        }
        if (!check(o.numResults() == data, "C5", "needs one result per data operand")) {
            return null;
        }
        for (int i = 0; i < data; i++) {
            check(
                    typeEq(o.operand(i).type(), o.result(i).type()),
                    dynamicRoot ? "C4" : "C3",
                    "result #" + i + " must have the operand type");
        }
        return null;
    }

    @Override
    public Void visitInfeed(InfeedOp op) {
        List<Type> rs = op.operation().resultTypes();
        if (check(!rs.isEmpty(), "C1", "needs at least one result")) {
            check(rs.getLast() instanceof TokenType, "C3", "the last result must be a token");
            for (int i = 0; i < rs.size() - 1; i++) {
                check(
                        rs.get(i) instanceof TensorType,
                        "C2",
                        "results before the token must be tensors");
            }
        }
        return null;
    }

    @Override
    public Void visitAsyncDone(AsyncDoneOp op) {
        if (op.future().type() instanceof FutureType f) {
            check(
                    typesEq(f.types(), op.operation().resultTypes()),
                    "C1",
                    "results must be the future types");
        }
        return null;
    }

    // ================================================================== send / recv / dynamic

    private boolean isHostTransfer() {
        return current.attribute("is_host_transfer").orElse(null)
                        instanceof com.qxotic.jota.stablehlo.attr.BoolAttr b
                && b.value();
    }

    @Override
    public Void visitSend(SendOp op) {
        long type = channelHandle("type");
        if (isHostTransfer()) {
            check(
                    type == 2,
                    "C5",
                    "channel_type should be DEVICE_TO_HOST when is_host_transfer is true");
        } else {
            check(
                    type == 1,
                    "C5",
                    "channel_type should be DEVICE_TO_DEVICE when is_host_transfer is false");
        }
        return null;
    }

    @Override
    public Void visitRecv(RecvOp op) {
        long type = channelHandle("type");
        if (isHostTransfer()) {
            check(
                    type == 3,
                    "C5",
                    "channel_type should be HOST_TO_DEVICE when is_host_transfer is true");
        } else {
            check(
                    type == 1,
                    "C5",
                    "channel_type should be DEVICE_TO_DEVICE when is_host_transfer is false");
        }
        List<Type> rs = op.operation().resultTypes();
        if (check(!rs.isEmpty(), "recv", "needs at least one result")) {
            check(rs.getLast() instanceof TokenType, "recv", "the last result must be a token");
            for (int i = 0; i < rs.size() - 1; i++) {
                check(
                        rs.get(i) instanceof TensorType,
                        "recv",
                        "results before the token must be tensors");
            }
        }
        return null;
    }

    @Override
    public Void visitCollectiveReduce(CollectiveReduceOp op) {
        checkGroups("C1", false, -1, false);
        Operation o = op.operation();
        boolean dynamicRoot = o.attribute("has_dynamic_root").isPresent();
        int data = o.numOperands() - (dynamicRoot ? 1 : 0);
        if (dynamicRoot) {
            if (!check(
                    o.numOperands() >= 2,
                    "C3",
                    "has_dynamic_root needs data operands and a root")) {
                return null;
            }
            TensorType root = tt(o.operand(o.numOperands() - 1));
            if (check(
                    root != null
                            && root.rank() == 1
                            && root.elementType() instanceof IntegerType it
                            && it.bitWidth() == 32,
                    "C3",
                    "the root operand must be a rank-1 i32 tensor")) {
                check(
                        dimEq(root.dim(0), data),
                        "C3",
                        "the root operand must have one element per data operand");
            }
        } else if (!check(data >= 1, "C3", "needs at least one data operand")) {
            return null;
        }
        for (int i = 0; i < data; i++) {
            TensorType x = tt(o.operand(i));
            if (x == null) {
                continue;
            }
            List<ElementType> es =
                    checkReducer(
                            op.computation(),
                            List.of(x.elementType()),
                            List.of(scalar(x.elementType())),
                            new long[0],
                            "C4");
            if (es == null) {
                return null;
            }
            if (i < o.numResults()) {
                check(shapeEq(x, tt(o.result(i))), "C3", "results must have the operand shapes");
            }
        }
        return null;
    }

    @Override
    public Void visitDynamicPad(DynamicPadOp op) {
        TensorType x = tt(op.operand());
        TensorType p = tt(op.paddingValue());
        TensorType r = tt(op.result());
        if (x == null || p == null || r == null) {
            return null;
        }
        check(
                elemEq(x.elementType(), p.elementType())
                        && elemEq(x.elementType(), r.elementType()),
                "C1",
                "operand, padding_value and result must have the same element type");
        List<Value> pads = List.of(op.edgePaddingLow(), op.edgePaddingHigh(), op.interiorPadding());
        for (Value v : pads) {
            TensorType t = tt(v);
            if (t != null
                    && !check(
                            t.rank() == 1 && dimEq(t.dim(0), x.rank()),
                            "C2",
                            "padding operands size must match operand rank (" + x.rank() + ")")) {
                return null;
            }
        }
        long[] lo = constant(op.edgePaddingLow());
        long[] hi = constant(op.edgePaddingHigh());
        long[] in = constant(op.interiorPadding());
        if (in != null
                && !check(
                        Arrays.stream(in).allMatch(v -> v >= 0),
                        "C3",
                        "interior_padding must be non-negative")) {
            return null;
        }
        if (lo != null && hi != null && in != null && r.rank() == x.rank()) {
            for (int i = 0; i < x.rank(); i++) {
                if (x.dim(i) != DYN) {
                    long want = x.dim(i) + lo[i] + Math.max(x.dim(i) - 1, 0) * in[i] + hi[i];
                    if (!check(
                            dimEq(r.dim(i), want),
                            "C4",
                            "expected output dimension at index " + i + " to equal " + want)) {
                        break;
                    }
                }
            }
        }
        return null;
    }

    // ================================================================== async / custom_call

    private static final Set<String> ASYNC_OPS =
            Set.of(
                    "stablehlo.all_reduce",
                    "stablehlo.all_gather",
                    "stablehlo.all_to_all",
                    "stablehlo.collective_permute",
                    "stablehlo.reduce_scatter",
                    "stablehlo.collective_broadcast",
                    "stablehlo.slice",
                    "stablehlo.dynamic_slice",
                    "stablehlo.dynamic_update_slice");

    @Override
    public Void visitAsyncStart(AsyncStartOp op) {
        Operation o = op.operation();
        if (!check(o.regions().size() == 1, "C1", "op requires one region")) {
            return null;
        }
        Block b = entry(o.region(0));
        if (!check(
                b != null && b.operations().size() == 2,
                "C1",
                "region must contain exactly one operation and a return")) {
            return null;
        }
        Operation inner = b.operations().getFirst();
        check(
                ASYNC_OPS.contains(inner.name()),
                "C2",
                "region must contain a collective or slice operation");
        List<Type> args = b.arguments().stream().map(Value::type).toList();
        if (check(
                args.size() == o.numOperands(),
                "C3",
                "number of operands ("
                        + o.numOperands()
                        + ") must match number of region arguments ("
                        + args.size()
                        + ")")) {
            check(
                    typesEq(o.operandTypes(), args),
                    "C3",
                    "operand types must match the region argument types");
        }
        List<Type> returned = b.terminator() == null ? List.of() : b.terminator().operandTypes();
        if (o.numResults() == 1 && o.result(0).type() instanceof FutureType f) {
            check(typesEq(f.types(), returned), "C4", "result must be future<" + returned + ">");
        }
        return null;
    }

    @Override
    public Void visitCustomCall(CustomCallOp op) {
        Operation o = op.operation();
        long api =
                op.apiVersion()
                        .map(
                                a ->
                                        a instanceof com.qxotic.jota.stablehlo.attr.IntegerAttr i
                                                ? i.value()
                                                : enumOrdinal(a))
                        .orElse(1L);
        op.backendConfig()
                .ifPresent(
                        cfg -> {
                            if (api == 4) {
                                check(
                                        cfg
                                                instanceof
                                                com.qxotic.jota.stablehlo.attr.DictionaryAttr,
                                        "custom_call",
                                        "backend_config for api_version API_VERSION_TYPED_FFI must"
                                                + " be a dictionary attribute");
                            } else {
                                check(
                                        cfg instanceof com.qxotic.jota.stablehlo.attr.StringAttr,
                                        "custom_call",
                                        "backend_config for api_version "
                                                + api
                                                + " must be a string attribute");
                            }
                        });
        var ol = op.operandLayouts();
        var rl = op.resultLayouts();
        if (ol.isPresent() || rl.isPresent()) {
            if (!check(
                    ol.isPresent() && rl.isPresent(),
                    "custom_call",
                    "layout attributes should be specified for either both operands and results or"
                            + " none")) {
                return null;
            }
            List<Type> results = o.resultTypes();
            if (results.size() == 1 && results.getFirst() instanceof TupleType t) {
                results = t.types(); // a single tuple result: one layout per element
            }
            checkLayouts(ol.get(), o.operandTypes(), "operand");
            checkLayouts(rl.get(), results, "result");
        }
        op.outputOperandAliases()
                .ifPresent(
                        a -> {
                            if (a instanceof ArrayAttr arr) {
                                for (Attribute alias : arr.elements()) {
                                    checkAlias(o, alias);
                                }
                            }
                        });
        return null;
    }

    private static long enumOrdinal(Attribute a) {
        return switch (enumName(a)) {
            case "API_VERSION_UNSPECIFIED" -> 0;
            case "API_VERSION_ORIGINAL" -> 1;
            case "API_VERSION_STATUS_RETURNING" -> 2;
            case "API_VERSION_STATUS_RETURNING_UNIFIED" -> 3;
            case "API_VERSION_TYPED_FFI" -> 4;
            default -> 1;
        };
    }

    private void checkLayouts(Attribute layouts, List<Type> types, String what) {
        if (!(layouts instanceof ArrayAttr arr)) {
            return;
        }
        if (!check(
                arr.elements().size() == types.size(),
                "custom_call",
                "number of "
                        + what
                        + "s must match the number of "
                        + what
                        + " layouts, "
                        + types.size()
                        + " != "
                        + arr.elements().size())) {
            return;
        }
        for (int i = 0; i < types.size(); i++) {
            if (!(arr.elements().get(i) instanceof DenseElementsAttr d)) {
                continue;
            }
            long[] layout = new long[(int) d.numElements()];
            for (int k = 0; k < layout.length; k++) {
                layout[k] = d.getLong(k);
            }
            if (!check(
                    !(types.get(i) instanceof TupleType),
                    "custom_call",
                    "tuple types are not fully supported with layout constraints yet")) {
                return;
            }
            if (!(types.get(i) instanceof TensorType t)) {
                check(
                        layout.length == 0,
                        "custom_call",
                        "only tensor types can have non-empty layout: " + what + " #" + i);
                continue;
            }
            check(
                    layout.length == t.rank() && unique(layout) && inRange(layout, 0, t.rank()),
                    "custom_call",
                    "incorrect layout "
                            + str(layout)
                            + " for "
                            + t.mlir()
                            + ", must be a permutation of [0, "
                            + t.rank()
                            + ")");
        }
    }

    private void checkAlias(Operation o, Attribute alias) {
        if (!(alias instanceof com.qxotic.jota.stablehlo.attr.DialectAttr d) || !d.hasParams()) {
            return;
        }
        long[] outIdx = d.param("output_tuple_indices").map(Verifier::longs).orElse(new long[0]);
        long operandIndex =
                d.param("operand_index")
                        .map(Verifier::longs)
                        .map(v -> v.length == 0 ? 0 : v[0])
                        .orElse(0L);
        long[] opIdx = d.param("operand_tuple_indices").map(Verifier::longs).orElse(new long[0]);
        if (!check(
                operandIndex >= 0 && operandIndex < o.numOperands(),
                "custom_call",
                "operand_index in output_operand_alias out of range [0, "
                        + o.numOperands()
                        + ")")) {
            return;
        }
        Type operand = walkTuple(o.operand((int) operandIndex).type(), opIdx);
        if (!check(
                operand != null,
                "custom_call",
                "operand_tuple_indices in the output_operand_alias attribute out of bounds")) {
            return;
        }
        Type output =
                o.numResults() == 1
                        ? walkTuple(o.result(0).type(), outIdx)
                        : outIdx.length > 0 && outIdx[0] >= 0 && outIdx[0] < o.numResults()
                                ? walkTuple(
                                        o.result((int) outIdx[0]).type(),
                                        Arrays.copyOfRange(outIdx, 1, outIdx.length))
                                : null;
        if (!check(
                output != null,
                "custom_call",
                "output_tuple_indices in the output_operand_alias attribute out of bounds")) {
            return;
        }
        check(
                typeEq(operand, output),
                "custom_call",
                "shapes mismatch in the output_operand_alias attribute");
    }

    private static Type walkTuple(Type t, long[] path) {
        for (long i : path) {
            if (!(t instanceof TupleType tuple) || i < 0 || i >= tuple.types().size()) {
                return null;
            }
            t = tuple.types().get((int) i);
        }
        return t;
    }

    private static long[] longs(Attribute a) {
        return switch (a) {
            case com.qxotic.jota.stablehlo.attr.IntegerAttr i -> new long[] {i.value()};
            case com.qxotic.jota.stablehlo.attr.DenseArrayAttr arr -> arr.values();
            case ArrayAttr arr ->
                    arr.elements().stream()
                            .mapToLong(e -> longs(e).length == 0 ? 0 : longs(e)[0])
                            .toArray();
            default -> new long[0];
        };
    }

    @Override
    public Void visitUniformQuantize(UniformQuantizeOp op) {
        TensorType x = tt(op.operand());
        TensorType r = tt(op.result());
        if (x == null || r == null) {
            return null;
        }
        check(shapeEq(x, r), "C1", "result shape must equal the operand shape");
        if (r.elementType() instanceof QuantizedType qr) {
            ElementType want =
                    x.elementType() instanceof QuantizedType qx
                            ? qx.expressedType()
                            : x.elementType();
            check(
                    qr.expressedType().equals(want),
                    "C2",
                    "expressed type of result expected to be " + want.mlir());
        }
        return null;
    }

    @Override
    public Void visitFma(FmaOp op) {
        List<Value> all = List.of(op.a(), op.b(), op.c(), op.operation().result(0));
        for (Value v : all) {
            TensorType t = tt(v);
            if (t != null
                    && !check(
                            t.elementType() instanceof FloatType,
                            "type",
                            "operands must be floating-point tensors, but got " + t.mlir())) {
                return null;
            }
        }
        for (Value v : all) {
            if (!check(
                    typeEq(v.type(), op.a().type()),
                    "type",
                    "requires compatible types for all operands and results")) {
                return null;
            }
        }
        return null;
    }
}
