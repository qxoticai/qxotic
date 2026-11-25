package com.qxotic.jota.stablehlo.text;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.AttrSyntax;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.bytecode.InherentAttributes;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Set;
import java.util.StringJoiner;

/**
 * Prints operations as MLIR text, in one of two forms.
 *
 * <p>{@link #print} uses the custom (pretty) assembly of the StableHLO, {@code func} and {@code
 * builtin} operations, as {@code stablehlo-opt} prints it:
 *
 * <pre>{@code
 * %0 = stablehlo.add %arg0, %arg1 : tensor<2xf32>
 * }</pre>
 *
 * The formats follow {@code StablehloOps.td} and the custom printers of {@code AssemblyFormat.cpp}
 * / {@code StablehloOps.cpp}. An operation of another dialect, or one whose attributes the pretty
 * form cannot spell exactly (a raw convolution layout, an empty {@code arg_attrs}...), prints in
 * generic form, so the output always parses back to a structurally equal program.
 *
 * <p>{@link #printGeneric} uses the MLIR generic form, which every MLIR tool accepts regardless of
 * custom assembly formats:
 *
 * <pre>{@code
 * %0 = "stablehlo.add"(%arg0, %arg1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
 * }</pre>
 *
 * Values are named as MLIR's printer names them ({@link SsaNames}): entry block arguments {@code
 * %argN}, op results {@code %N} ({@code %N:k} / {@code %N#i} for several), and in the pretty form
 * the dialects' names ({@code %cst} / {@code %c} constants, {@code %iterArg} while arguments,
 * {@code %output_state, %output}...) with MLIR's {@code _<n>} suffixes for repeated names.
 */
public final class MlirPrinter {

    /** Commutative binary ops that {@code stablehlo.reduce ... applies <op>} may name. */
    private static final Set<String> COMPACT_REDUCERS =
            Set.of(
                    "stablehlo.add",
                    "stablehlo.multiply",
                    "stablehlo.maximum",
                    "stablehlo.minimum",
                    "stablehlo.and",
                    "stablehlo.or",
                    "stablehlo.xor");

    /** Ops printed {@code op %a, %b {attrs} : type} with {@code SameOperandsAndResultType}. */
    private static final Set<String> SAME_TYPE_OPS =
            Set.of(
                    "abs",
                    "cbrt",
                    "ceil",
                    "convert",
                    "count_leading_zeros",
                    "cosine",
                    "exponential",
                    "exponential_minus_one",
                    "exp2",
                    "floor",
                    "imag",
                    "log",
                    "log_plus_one",
                    "log2",
                    "logistic",
                    "not",
                    "negate",
                    "popcnt",
                    "real",
                    "round_nearest_afz",
                    "round_nearest_even",
                    "rsqrt",
                    "sign",
                    "sine",
                    "sqrt",
                    "tan",
                    "tanh",
                    "add",
                    "atan2",
                    "divide",
                    "maximum",
                    "minimum",
                    "multiply",
                    "power",
                    "remainder",
                    "shift_left",
                    "shift_right_arithmetic",
                    "shift_right_logical",
                    "subtract",
                    "and",
                    "or",
                    "xor",
                    "clamp",
                    "fma",
                    "uniform_quantize",
                    "uniform_dequantize",
                    "after_all");

    /** Ops printed {@code op %a, %b {attrs} : (A, B) -> R}. */
    private static final Set<String> FUNCTIONAL_OPS =
            Set.of(
                    "is_finite",
                    "bitcast_convert",
                    "reshape",
                    "dynamic_reshape",
                    "dynamic_update_slice",
                    "real_dynamic_slice",
                    "dynamic_pad");

    /** Ops printed {@code op {attrs} : type} (no operands, one result). */
    private static final Set<String> NULLARY_OPS =
            Set.of("create_token", "replica_id", "partition_id");

    private final boolean pretty;
    private final StringBuilder out = new StringBuilder();
    private final Deque<String> defaultDialect = new ArrayDeque<>();
    private final SsaNames names;

    private MlirPrinter(Operation top, boolean pretty) {
        this.pretty = pretty;
        this.names = SsaNames.of(top, pretty);
        defaultDialect.push("");
    }

    /** Pretty form of a program. */
    public static String print(Program module) {
        return print(module.operation());
    }

    /** Pretty form of an operation (and the operations nested in it). */
    public static String print(Operation operation) {
        MlirPrinter p = new MlirPrinter(operation, true);
        p.operation(operation, 0);
        return p.out.toString();
    }

    /** Generic form of a program. */
    public static String printGeneric(Program module) {
        return printGeneric(module.operation());
    }

    /** Generic form of an operation (and the operations nested in it). */
    public static String printGeneric(Operation operation) {
        MlirPrinter p = new MlirPrinter(operation, false);
        p.operation(operation, 0);
        return p.out.toString();
    }

    // ================================================================== common

    private void indent(int n) {
        out.append("  ".repeat(n));
    }

    private String name(Value v) {
        String n = names.name(v);
        if (n == null) {
            throw new IllegalStateException("value used before its definition: " + v);
        }
        return n;
    }

    private String names(List<Value> values) {
        StringJoiner j = new StringJoiner(", ");
        for (Value v : values) {
            j.add(name(v));
        }
        return j.toString();
    }

    private static String types(List<Type> types) {
        StringJoiner j = new StringJoiner(", ");
        for (Type t : types) {
            j.add(t.mlir());
        }
        return j.toString();
    }

    private void operation(Operation op, int depth) {
        indent(depth);
        if (op.numResults() > 0) {
            out.append(names.resultList(op)).append(" = ");
        }
        if (!pretty || !pretty(op, depth)) {
            generic(op, depth);
        }
        out.append('\n');
    }

    /**
     * {@code "name"(operands) <{properties}> (regions) {attributes} : type}. As in MLIR, the
     * inherent attributes of a registered op are properties, and both dictionaries are sorted.
     */
    private void generic(Operation op, int depth) {
        List<NamedAttribute> properties = new ArrayList<>(op.properties());
        List<NamedAttribute> attributes = new ArrayList<>();
        Set<String> inherent = InherentAttributes.of(op.name());
        for (NamedAttribute a : op.attributes()) {
            boolean taken = properties.stream().anyMatch(p -> p.name().equals(a.name()));
            (inherent != null && inherent.contains(a.name()) && !taken ? properties : attributes)
                    .add(a);
        }
        properties.sort(java.util.Comparator.comparing(NamedAttribute::name));
        attributes.sort(java.util.Comparator.comparing(NamedAttribute::name));
        out.append('"').append(op.name()).append("\"(").append(names(op.operands())).append(')');
        if (!properties.isEmpty()) {
            out.append(" <");
            dict(properties);
            out.append('>');
        }
        if (!op.regions().isEmpty()) {
            defaultDialect.push("");
            out.append(" (");
            for (int i = 0; i < op.regions().size(); i++) {
                if (i > 0) {
                    out.append(", ");
                }
                region(op.region(i), depth, true);
            }
            out.append(')');
            defaultDialect.pop();
        }
        if (!attributes.isEmpty()) {
            out.append(' ');
            dict(attributes);
        }
        out.append(" : ").append(functionalType(op));
    }

    private void dict(List<NamedAttribute> entries) {
        out.append('{');
        for (int i = 0; i < entries.size(); i++) {
            out.append(i > 0 ? ", " : "").append(entries.get(i));
        }
        out.append('}');
    }

    /**
     * Prints {@code { ops }}; the entry block label and arguments are printed when {@code
     * entryArgs} is set (or required), otherwise the entry arguments must already be named.
     */
    private void region(Region region, int depth, boolean entryArgs) {
        out.append("{\n");
        int bb = 0;
        for (Block block : region.blocks()) {
            boolean label =
                    bb > 0
                            || region.blocks().size() > 1
                            || (entryArgs
                                    && (!block.arguments().isEmpty()
                                            || block.operations().isEmpty()));
            if (label) {
                indent(depth);
                out.append("^bb").append(bb);
                if (!block.arguments().isEmpty()) {
                    out.append('(');
                    for (int i = 0; i < block.arguments().size(); i++) {
                        Value a = block.argument(i);
                        String n = name(a);
                        out.append(i > 0 ? ", " : "")
                                .append(n)
                                .append(": ")
                                .append(a.type().mlir());
                    }
                    out.append(')');
                }
                out.append(":\n");
            }
            bb++;
            for (Operation op : block.operations()) {
                operation(op, depth + 1);
            }
        }
        indent(depth);
        out.append('}');
    }

    private static String functionalType(Operation op) {
        return new FunctionType(op.operandTypes(), op.resultTypes()).mlir();
    }

    /** {@code SameOperandsAndResultType}: one type when every operand has the result type. */
    private static String sameType(Operation op) {
        Type result = op.result(0).type();
        for (Type t : op.operandTypes()) {
            if (!t.equals(result)) {
                return functionalType(op);
            }
        }
        return result.mlir();
    }

    /** The op name as printed: the enclosing default dialect ({@code func}) is dropped. */
    private String opName(Operation op) {
        String d = defaultDialect.peek();
        if (!d.isEmpty() && op.name().startsWith(d + ".")) {
            return op.name().substring(d.length() + 1);
        }
        return op.name();
    }

    /** {@code " {a = 1, ...}"} with the attributes not spelled by the pretty form. */
    private static String attrDict(Operation op, Set<String> elided) {
        List<NamedAttribute> rest = new ArrayList<>();
        for (NamedAttribute a : op.allAttributes()) {
            if (!elided.contains(a.name())) {
                rest.add(a);
            }
        }
        if (rest.isEmpty()) {
            return "";
        }
        StringJoiner j = new StringJoiner(", ", " {", "}");
        for (NamedAttribute a : rest) {
            j.add(a.toString());
        }
        return j.toString();
    }

    /** True when no attribute name occurs twice (the pretty forms merge both dictionaries). */
    private static boolean distinctAttributeNames(Operation op) {
        List<NamedAttribute> all = op.allAttributes();
        return all.stream().map(NamedAttribute::name).distinct().count() == all.size();
    }

    // ================================================================== attribute spellings

    private static Attribute attr(Operation op, String name) {
        return op.attribute(name).orElse(null);
    }

    /** {@code [1, 2]} for an i64 dense array, else null. */
    private static String i64List(Attribute a) {
        if (a instanceof DenseArrayAttr d && d.elementType().equals(IntegerType.I64)) {
            return longs(d.values());
        }
        return null;
    }

    private static String longs(long[] values) {
        StringJoiner j = new StringJoiner(", ", "[", "]");
        for (long v : values) {
            j.add(Long.toString(v));
        }
        return j.toString();
    }

    /** The bare literal of an integer attribute of the given type, else null. */
    private static String intOf(Attribute a, IntegerType type) {
        return a instanceof IntegerAttr i && i.type().equals(type) ? i.literal() : null;
    }

    /** The bare value of a {@code #stablehlo<kind VALUE>} enum, else null. */
    private static String enumOf(Attribute a, String kind) {
        return a instanceof EnumAttr e && e.dialect().equals("stablehlo") && e.kind().equals(kind)
                ? e.value()
                : null;
    }

    /** An enum keyword value ({@code type = FFT}). */
    private static String enumKeyword(Attribute a, String kind) {
        return enumOf(a, kind);
    }

    // ================================================================== pretty forms

    /**
     * Prints the pretty form of {@code op} and returns true, or prints nothing and returns false.
     */
    private boolean pretty(Operation op, int depth) {
        if (!distinctAttributeNames(op)) {
            return false;
        }
        String name = op.name();
        if (name.equals("builtin.module")) {
            return module(op, depth);
        }
        if (name.equals("func.func")) {
            return func(op, depth);
        }
        if (name.equals("func.call")) {
            return call(op);
        }
        if (name.equals("func.return") || name.equals("stablehlo.return")) {
            return returnOp(op);
        }
        if (op.dialect().equals("chlo")) {
            return op.regions().isEmpty() && chlo(op);
        }
        if (!op.dialect().equals("stablehlo")) {
            return false;
        }
        String m = op.mnemonic();
        if (!op.regions().isEmpty() && !m.equals("while") && !m.equals("reduce")) {
            return false;
        }
        if (SAME_TYPE_OPS.contains(m) || FUNCTIONAL_OPS.contains(m)) {
            if (op.numResults() != 1 || (op.numOperands() == 0 && !m.equals("after_all"))) {
                return false;
            }
            if (op.numOperands() == 0) { // MLIR keeps the space of the empty operand list
                out.append(op.name()).append(' ').append(attrDict(op, Set.of()));
                out.append(" : ").append(sameType(op));
                return true;
            }
            emit(
                    op,
                    names(op.operands()),
                    Set.of(),
                    FUNCTIONAL_OPS.contains(m) ? functionalType(op) : sameType(op));
            return true;
        }
        if (NULLARY_OPS.contains(m)) {
            if (op.numOperands() != 0 || op.numResults() != 1) {
                return false;
            }
            out.append(name).append(attrDict(op, Set.of())).append(" : ");
            out.append(op.result(0).type().mlir());
            return true;
        }
        return switch (m) {
            case "constant" -> constant(op);
            case "iota" -> iota(op);
            case "compare" -> compare(op);
            case "slice" -> slice(op);
            case "get_tuple_element" -> getTupleElement(op);
            case "tuple" -> tuple(op);
            case "select" -> select(op);
            case "complex" -> complex(op);
            case "optimization_barrier" -> optimizationBarrier(op);
            case "custom_call" -> customCall(op);
            case "composite" -> composite(op);
            case "dot" -> dot(op, false);
            case "dot_general" -> dot(op, true);
            case "convolution" -> convolution(op);
            case "while" -> whileOp(op, depth);
            case "reduce" -> reduce(op, depth);
            case "broadcast" -> keywords(op, 1, false, kw("sizes", "broadcast_sizes", I64_LIST));
            case "broadcast_in_dim" ->
                    keywords(op, 1, false, kw("dims", "broadcast_dimensions", I64_LIST));
            case "dynamic_broadcast_in_dim" ->
                    keywords(op, 2, false, kw("dims", "broadcast_dimensions", I64_LIST));
            case "transpose" -> keywords(op, 1, false, kw("dims", "permutation", I64_LIST));
            case "reverse" -> keywords(op, 1, true, kw("dims", "dimensions", I64_LIST));
            case "dynamic_iota" -> keywords(op, 1, false, kw("dim", "iota_dimension", I64));
            case "get_dimension_size" -> keywords(op, 1, false, kw("dim", "dimension", I64));
            case "set_dimension_size" -> keywords(op, 2, false, kw("dim", "dimension", I64));
            case "concatenate" -> keywords(op, -1, false, kw("dim", "dimension", I64));
            case "dynamic_slice" -> keywords(op, -1, false, kw("sizes", "slice_sizes", I64_LIST));
            case "pad" ->
                    keywords(
                            op,
                            2,
                            false,
                            kw("low", "edge_padding_low", I64_LIST),
                            kw("high", "edge_padding_high", I64_LIST),
                            kw("interior", "interior_padding", I64_LIST));
            case "fft" ->
                    keywords(
                            op,
                            1,
                            false,
                            kw("type", "fft_type", a -> enumKeyword(a, "fft_type")),
                            kw("length", "fft_length", I64_LIST));
            case "rng" ->
                    keywords(
                            op,
                            3,
                            false,
                            kw(
                                    "distribution",
                                    "rng_distribution",
                                    a -> enumKeyword(a, "rng_distribution")));
            case "rng_bit_generator" ->
                    keywords(
                            op,
                            1,
                            false,
                            kw("algorithm", "rng_algorithm", a -> enumKeyword(a, "rng_algorithm")));
            case "einsum" -> keywords(op, 2, false, kw("config", "einsum_config", STRING));
            case "unary_einsum" -> keywords(op, 1, false, kw("config", "einsum_config", STRING));
            case "cholesky" -> cholesky(op);
            case "reduce_precision" -> reducePrecision(op);
            default -> false;
        };
    }

    /** {@code name operands{attrs} : sig}. */
    private void emit(Operation op, String operands, Set<String> elided, String sig) {
        out.append(op.name());
        if (!operands.isEmpty()) {
            out.append(' ').append(operands);
        }
        out.append(attrDict(op, elided)).append(" : ").append(sig);
    }

    private interface Spelling {
        /** The pretty spelling of an attribute, or null when the pretty form cannot hold it. */
        String of(Attribute a);
    }

    private static final Spelling I64_LIST = MlirPrinter::i64List;
    private static final Spelling I64 = a -> intOf(a, IntegerType.I64);
    private static final Spelling STRING =
            a -> a instanceof StringAttr s ? AttrSyntax.quote(s.value()) : null;

    private record Keyword(String keyword, String attribute, Spelling spelling) {}

    private static Keyword kw(String keyword, String attribute, Spelling spelling) {
        return new Keyword(keyword, attribute, spelling);
    }

    /**
     * {@code op %a, ..., kw = v, ... {attrs} : sig}: {@code operands} operands (-1: all), every
     * keyword attribute present and spellable.
     */
    private boolean keywords(Operation op, int operands, boolean sameType, Keyword... kws) {
        if (op.numResults() == 0
                || (sameType && op.numResults() != 1)
                || (operands >= 0 && op.numOperands() != operands)) {
            return false;
        }
        StringJoiner parts = new StringJoiner(", ");
        for (Value v : op.operands()) {
            parts.add(name(v));
        }
        Set<String> elided = new java.util.HashSet<>();
        for (Keyword k : kws) {
            Attribute a = attr(op, k.attribute);
            String s = a == null ? null : k.spelling.of(a);
            if (s == null) {
                return false;
            }
            parts.add(k.keyword + " = " + s);
            elided.add(k.attribute);
        }
        out.append(op.name()).append(' ').append(parts);
        out.append(attrDict(op, elided)).append(" : ");
        out.append(sameType ? sameType(op) : functionalType(op));
        return true;
    }

    /** {@code builtin.module [@name] [attributes {...}] { ... }}. */
    private boolean module(Operation op, int depth) {
        if (op.regions().size() != 1
                || op.region(0).blocks().size() != 1
                || !op.region(0).block().arguments().isEmpty()) {
            return false;
        }
        out.append("module");
        Set<String> elided = Set.of();
        if (attr(op, "sym_name") instanceof StringAttr s) {
            out.append(' ').append(SymbolRefAttr.of(s.value()));
            elided = Set.of("sym_name");
        }
        String rest = attrDict(op, elided);
        if (!rest.isEmpty()) {
            out.append(" attributes").append(rest);
        }
        out.append(' ');
        defaultDialect.push("");
        region(op.region(0), depth, false);
        defaultDialect.pop();
        return true;
    }

    /** Per-argument dictionaries of {@code arg_attrs} / {@code res_attrs}, or null. */
    private static List<DictionaryAttr> argDicts(Operation op, String name, int count) {
        Attribute a = attr(op, name);
        if (a == null) {
            return java.util.Collections.nCopies(count, DictionaryAttr.EMPTY);
        }
        if (!(a instanceof ArrayAttr arr) || arr.elements().size() != count) {
            return null;
        }
        List<DictionaryAttr> dicts = new ArrayList<>();
        boolean any = false;
        for (Attribute e : arr.elements()) {
            if (!(e instanceof DictionaryAttr d)) {
                return null;
            }
            any |= !d.isEmpty();
            dicts.add(d);
        }
        return any ? dicts : null; // all-empty arrays would not survive a re-parse
    }

    private static final Set<String> FUNC_PROPS =
            Set.of("sym_name", "function_type", "sym_visibility", "arg_attrs", "res_attrs");

    /** {@code func.func [visibility] @name(%arg0: T {..}) -> (R {..}) [attributes {..}] {..}}. */
    private boolean func(Operation op, int depth) {
        if (!(attr(op, "sym_name") instanceof StringAttr symbol)
                || !(attr(op, "function_type") instanceof TypeAttr ta)
                || !(ta.type() instanceof FunctionType fn)
                || op.regions().size() != 1) {
            return false;
        }
        String visibility = null;
        if (attr(op, "sym_visibility") != null) {
            if (!(attr(op, "sym_visibility") instanceof StringAttr v)
                    || !Set.of("public", "private", "nested").contains(v.value())) {
                return false;
            }
            visibility = v.value();
        }
        List<DictionaryAttr> argAttrs = argDicts(op, "arg_attrs", fn.inputs().size());
        List<DictionaryAttr> resAttrs = argDicts(op, "res_attrs", fn.results().size());
        if (argAttrs == null || resAttrs == null) {
            return false;
        }
        Region body = op.region(0);
        Block entry = null;
        if (!body.isEmpty()) {
            if (body.blocks().size() != 1) {
                return false;
            }
            entry = body.block();
            if (!entry.arguments().stream().map(Value::type).toList().equals(fn.inputs())) {
                return false;
            }
        }
        out.append(opName(op)).append(' ');
        if (visibility != null) {
            out.append(visibility).append(' ');
        }
        out.append(SymbolRefAttr.of(symbol.value())).append('(');
        for (int i = 0; i < fn.inputs().size(); i++) {
            if (i > 0) {
                out.append(", ");
            }
            if (entry != null) {
                out.append(name(entry.argument(i))).append(": ");
            }
            out.append(fn.inputs().get(i).mlir());
            if (!argAttrs.get(i).isEmpty()) {
                out.append(' ').append(argAttrs.get(i));
            }
        }
        out.append(')');
        List<Type> results = fn.results();
        if (!results.isEmpty()) {
            out.append(" -> ");
            boolean parens =
                    results.size() > 1
                            || !resAttrs.getFirst().isEmpty()
                            || results.getFirst() instanceof FunctionType;
            if (parens) {
                out.append('(');
            }
            for (int i = 0; i < results.size(); i++) {
                if (i > 0) {
                    out.append(", ");
                }
                out.append(results.get(i).mlir());
                if (!resAttrs.get(i).isEmpty()) {
                    out.append(' ').append(resAttrs.get(i));
                }
            }
            if (parens) {
                out.append(')');
            }
        }
        String rest = attrDict(op, FUNC_PROPS);
        if (!rest.isEmpty()) {
            out.append(" attributes").append(rest);
        }
        if (entry != null) {
            out.append(' ');
            defaultDialect.push("func");
            region(body, depth, false);
            defaultDialect.pop();
        }
        return true;
    }

    /** {@code call @f(%a) {..} : (A) -> R}. */
    private boolean call(Operation op) {
        if (!(attr(op, "callee") instanceof SymbolRefAttr callee) || callee.path().size() != 1) {
            return false;
        }
        out.append(opName(op)).append(' ').append(callee).append('(');
        out.append(names(op.operands())).append(')');
        out.append(attrDict(op, Set.of("callee"))).append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code return %a, %b {..} : A, B}. */
    private boolean returnOp(Operation op) {
        if (op.numResults() != 0) {
            return false;
        }
        out.append(opName(op));
        if (op.numOperands() > 0) {
            out.append(' ').append(names(op.operands()));
        }
        out.append(attrDict(op, Set.of()));
        if (op.numOperands() > 0) {
            out.append(" : ").append(types(op.operandTypes()));
        }
        return true;
    }

    /** {@code stablehlo.constant {..} dense<..> : T} when the value has the result type. */
    private boolean constant(Operation op) {
        if (op.numOperands() != 0
                || op.numResults() != 1
                || !(attr(op, "value") instanceof DenseElementsAttr value)
                || !value.type().equals(op.result(0).type())) {
            return false;
        }
        out.append(op.name()).append(attrDict(op, Set.of("value"))).append(' ').append(value);
        return true;
    }

    /** {@code stablehlo.iota dim = 0 {..} : T}. */
    private boolean iota(Operation op) {
        String dim = intOf(attr(op, "iota_dimension"), IntegerType.I64);
        if (op.numOperands() != 0 || op.numResults() != 1 || dim == null) {
            return false;
        }
        out.append(op.name()).append(" dim = ").append(dim);
        out.append(attrDict(op, Set.of("iota_dimension"))).append(" : ");
        out.append(op.result(0).type().mlir());
        return true;
    }

    /** {@code CHLO_UnaryElementwiseOp}s of {@code ChloOps.td}: {@code chlo.erf %x : T -> R}. */
    private static final Set<String> CHLO_UNARY =
            Set.of(
                    "_asin_acos_kernel",
                    "acos",
                    "acosh",
                    "asin",
                    "asinh",
                    "atan",
                    "atanh",
                    "bessel_i1e",
                    "conj",
                    "cosh",
                    "digamma",
                    "erf",
                    "erf_inv",
                    "erfc",
                    "is_inf",
                    "is_neg_inf",
                    "is_pos_inf",
                    "lgamma",
                    "sinh",
                    "square",
                    "tan");

    /** CHLO binary ops printed {@code chlo.next_after %x, %y : A, B -> R}. */
    private static final Set<String> CHLO_BINARY =
            Set.of("next_after", "polygamma", "zeta", "mulhi");

    /**
     * The assembly formats of {@code ChloOps.td}: unary and binary element-wise ops, {@code
     * broadcast_*} ({@code chlo.broadcast_add %a, %b : (A, B) -> R}) and {@code top_k}; the other
     * CHLO ops print in generic form.
     */
    private boolean chlo(Operation op) {
        String m = op.mnemonic();
        int arity = op.numOperands();
        if (m.equals("top_k")) {
            return topK(op);
        }
        if (op.numResults() != 1) {
            return false;
        }
        String result = op.result(0).type().mlir();
        if (CHLO_UNARY.contains(m) && arity == 1) {
            emit(op, names(op.operands()), Set.of(), op.operand(0).type().mlir() + " -> " + result);
        } else if (CHLO_BINARY.contains(m) && arity == 2) {
            emit(op, names(op.operands()), Set.of(), types(op.operandTypes()) + " -> " + result);
        } else if (m.startsWith("broadcast_") && arity == (m.equals("broadcast_select") ? 3 : 2)) {
            emit(op, names(op.operands()), Set.of(), functionalType(op));
        } else {
            return false;
        }
        return true;
    }

    /** {@code chlo.top_k(%x, k = 2[, is_stable = true]) {..} : T -> (V, I)}. */
    private boolean topK(Operation op) {
        String k = I64.of(attr(op, "k"));
        Attribute stable = attr(op, "is_stable");
        if (op.numOperands() != 1
                || op.numResults() != 2
                || k == null
                || (stable != null && !(stable instanceof BoolAttr))) {
            return false;
        }
        out.append(op.name()).append('(').append(name(op.operand(0))).append(", k = ").append(k);
        if (stable != null) {
            out.append(", is_stable = ").append(stable);
        }
        out.append(')').append(attrDict(op, Set.of("k", "is_stable")));
        out.append(" : ").append(op.operand(0).type().mlir());
        out.append(" -> (").append(types(op.resultTypes())).append(')');
        return true;
    }

    /** {@code stablehlo.compare LT, %a, %b, FLOAT {..} : (A, B) -> R}. */
    private boolean compare(Operation op) {
        String dir = enumOf(attr(op, "comparison_direction"), "comparison_direction");
        Attribute typeAttr = attr(op, "compare_type");
        String type = typeAttr == null ? null : enumOf(typeAttr, "comparison_type");
        if (op.numOperands() != 2
                || op.numResults() != 1
                || dir == null
                || (typeAttr != null && type == null)) {
            return false;
        }
        out.append(op.name()).append(' ').append(dir).append(", ");
        out.append(names(op.operands()));
        if (type != null) {
            out.append(", ").append(type);
        }
        out.append(attrDict(op, Set.of("comparison_direction", "compare_type")));
        out.append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code stablehlo.slice %x [0:2, 1:4:2] {..} : (A) -> R}. */
    private boolean slice(Operation op) {
        String s = i64List(attr(op, "start_indices"));
        String l = i64List(attr(op, "limit_indices"));
        String st = i64List(attr(op, "strides"));
        if (op.numOperands() != 1 || op.numResults() != 1 || s == null || l == null || st == null) {
            return false;
        }
        long[] start = ((DenseArrayAttr) attr(op, "start_indices")).values();
        long[] limit = ((DenseArrayAttr) attr(op, "limit_indices")).values();
        long[] stride = ((DenseArrayAttr) attr(op, "strides")).values();
        if (start.length != limit.length || start.length != stride.length) {
            return false;
        }
        StringJoiner ranges = new StringJoiner(", ", "[", "]");
        for (int i = 0; i < start.length; i++) {
            ranges.add(start[i] + ":" + limit[i] + (stride[i] != 1 ? ":" + stride[i] : ""));
        }
        out.append(op.name()).append(' ').append(name(op.operand(0))).append(' ').append(ranges);
        out.append(attrDict(op, Set.of("start_indices", "limit_indices", "strides")));
        out.append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code stablehlo.get_tuple_element %t[0] {..} : (tuple<..>) -> T}. */
    private boolean getTupleElement(Operation op) {
        String index = intOf(attr(op, "index"), IntegerType.I32);
        if (op.numOperands() != 1 || op.numResults() != 1 || index == null) {
            return false;
        }
        out.append(op.name()).append(' ').append(name(op.operand(0)));
        out.append('[').append(index).append(']');
        out.append(attrDict(op, Set.of("index"))).append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code stablehlo.tuple %a, %b {..} : tuple<A, B>}. */
    private boolean tuple(Operation op) {
        if (op.numResults() != 1
                || !(op.result(0).type() instanceof TupleType tt)
                || !tt.types().equals(op.operandTypes())) {
            return false;
        }
        emit(op, names(op.operands()), Set.of(), tt.mlir());
        return true;
    }

    /** {@code stablehlo.select %p, %a, %b {..} : P, T} (or a function type). */
    private boolean select(Operation op) {
        if (op.numOperands() != 3 || op.numResults() != 1) {
            return false;
        }
        Type result = op.result(0).type();
        String sig =
                op.operand(1).type().equals(result) && op.operand(2).type().equals(result)
                        ? op.operand(0).type().mlir() + ", " + result.mlir()
                        : functionalType(op);
        emit(op, names(op.operands()), Set.of(), sig);
        return true;
    }

    /** {@code stablehlo.complex %re, %im {..} : tensor<..xcomplex<..>>} (or a function type). */
    private boolean complex(Operation op) {
        if (op.numOperands() != 2 || op.numResults() != 1) {
            return false;
        }
        Type result = op.result(0).type();
        String sig = functionalType(op);
        if (result instanceof TensorType tt && tt.elementType() instanceof ComplexType ct) {
            Type part = tt.withElementType(ct.elementType());
            if (op.operand(0).type().equals(part) && op.operand(1).type().equals(part)) {
                sig = result.mlir();
            }
        }
        emit(op, names(op.operands()), Set.of(), sig);
        return true;
    }

    /** {@code stablehlo.optimization_barrier {..} %a, %b : A, B}. */
    private boolean optimizationBarrier(Operation op) {
        if (!op.operandTypes().equals(op.resultTypes())) {
            return false;
        }
        out.append(op.name()).append(attrDict(op, Set.of()));
        if (op.numOperands() == 0) {
            out.append(" ()");
        } else {
            out.append(' ').append(names(op.operands()));
            out.append(" : ").append(types(op.operandTypes()));
        }
        return true;
    }

    /** {@code stablehlo.custom_call @target(%a) {..} : (A) -> R}. */
    private boolean customCall(Operation op) {
        if (!(attr(op, "call_target_name") instanceof StringAttr target)
                || target.value().isEmpty()) {
            return false;
        }
        out.append(op.name()).append(' ').append(SymbolRefAttr.of(target.value()));
        out.append('(').append(names(op.operands())).append(')');
        out.append(attrDict(op, Set.of("call_target_name")));
        out.append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code stablehlo.composite "name" %a {..} : (A) -> R} (no composite regions). */
    private boolean composite(Operation op) {
        if (!(attr(op, "name") instanceof StringAttr n)) {
            return false;
        }
        out.append(op.name()).append(' ').append(AttrSyntax.quote(n.value()));
        if (op.numOperands() > 0) {
            out.append(' ').append(names(op.operands()));
        }
        out.append(attrDict(op, Set.of("name"))).append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code , precision = [DEFAULT, HIGH]}, "" when absent, null when not spellable. */
    private static String precision(Operation op) {
        Attribute a = attr(op, "precision_config");
        if (a == null) {
            return "";
        }
        if (!(a instanceof ArrayAttr arr)) {
            return null;
        }
        StringJoiner j = new StringJoiner(", ", ", precision = [", "]");
        for (Attribute e : arr.elements()) {
            String v = enumOf(e, "precision");
            if (v == null) {
                return null;
            }
            j.add(v);
        }
        return j.toString();
    }

    /**
     * {@code stablehlo.dot %a, %b[, precision = [..]] {..} : sig}; {@code stablehlo.dot_general %a,
     * %b, [ext,] [batching_dims = [..] x [..],] contracting_dims = [..] x [..][, precision =
     * [..]][, algorithm = <..>] {..} : sig}.
     */
    private boolean dot(Operation op, boolean general) {
        String precision = precision(op);
        if (precision == null
                || op.numResults() != 1
                || op.numOperands() < 2
                || (!general && op.numOperands() != 2)) {
            return false;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(op.name()).append(' ').append(name(op.operand(0)));
        sb.append(", ").append(name(op.operand(1)));
        Set<String> elided = new java.util.HashSet<>(Set.of("precision_config"));
        if (general) {
            if (op.numOperands() > 2) {
                sb.append(", [").append(names(op.operands().subList(2, op.numOperands())));
                sb.append(']');
            }
            Attribute dims = attr(op, "dot_dimension_numbers");
            DotDimensionNumbers d;
            try {
                d = DotDimensionNumbers.from(dims);
            } catch (RuntimeException e) {
                return false;
            }
            if (!d.toAttribute().equals(dims)) {
                return false;
            }
            sb.append(", ");
            if (d.lhsBatchingDimensions().length > 0 || d.rhsBatchingDimensions().length > 0) {
                sb.append("batching_dims = ").append(longs(d.lhsBatchingDimensions()));
                sb.append(" x ").append(longs(d.rhsBatchingDimensions())).append(", ");
            }
            sb.append("contracting_dims = ").append(longs(d.lhsContractingDimensions()));
            sb.append(" x ").append(longs(d.rhsContractingDimensions()));
            elided.add("dot_dimension_numbers");
        }
        sb.append(precision);
        if (general && attr(op, "algorithm") != null) {
            String prefix = "#stablehlo.dot_algorithm";
            if (!(attr(op, "algorithm") instanceof DialectAttr alg)
                    || !alg.dialect().equals("stablehlo")
                    || !alg.mnemonic().equals("dot_algorithm")
                    || !alg.toString().startsWith(prefix)) {
                return false;
            }
            sb.append(", algorithm = ").append(alg.toString().substring(prefix.length()));
            elided.add("algorithm");
        }
        out.append(sb).append(attrDict(op, elided)).append(" : ").append(functionalType(op));
        return true;
    }

    /** {@code [[lo, hi], ...]} of an {@code Nx2xi64} padding, else null. */
    private static String padding(Attribute a) {
        if (!(a instanceof DenseElementsAttr d)
                || !(d.type().elementType().equals(IntegerType.I64))
                || d.type().rank() != 2
                || d.type().dim(1) != 2) {
            return null;
        }
        StringJoiner j = new StringJoiner(", ", "[", "]");
        for (long i = 0; i < d.numElements(); i += 2) {
            j.add("[" + d.getLong(i) + ", " + d.getLong(i + 1) + "]");
        }
        return j.toString();
    }

    /**
     * {@code stablehlo.convolution(%l, %r) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window =
     * {stride = [..], pad = [[..]], lhs_dilate = [..], rhs_dilate = [..], reverse = [..]} {..} :
     * sig}.
     */
    private boolean convolution(Operation op) {
        if (op.numOperands() != 2 || op.numResults() != 1) {
            return false;
        }
        Attribute dims = attr(op, "dimension_numbers");
        ConvDimensionNumbers d;
        try {
            d = ConvDimensionNumbers.from(dims);
        } catch (RuntimeException e) {
            return false;
        }
        if (!d.toAttribute().equals(dims)) {
            return false;
        }
        StringJoiner window = new StringJoiner(", ", "{", "}");
        String[][] fields = {
            {"stride", "window_strides"},
            {"pad", "padding"},
            {"lhs_dilate", "lhs_dilation"},
            {"rhs_dilate", "rhs_dilation"},
            {"reverse", "window_reversal"}
        };
        Set<String> elided = new java.util.HashSet<>(Set.of("dimension_numbers"));
        for (String[] f : fields) {
            Attribute a = attr(op, f[1]);
            if (a == null) {
                continue;
            }
            String s =
                    switch (f[0]) {
                        case "pad" -> padding(a);
                        case "reverse" ->
                                a instanceof DenseArrayAttr b
                                                && b.elementType().equals(IntegerType.I1)
                                        ? booleans(b.booleans())
                                        : null;
                        default -> i64List(a);
                    };
            if (s == null) {
                return false;
            }
            window.add(f[0] + " = " + s);
            elided.add(f[1]);
        }
        out.append(op.name()).append('(').append(names(op.operands())).append(')');
        out.append(" dim_numbers = ").append(d.compact()).append(", window = ").append(window);
        out.append(attrDict(op, elided)).append(" : ").append(functionalType(op));
        return true;
    }

    private static String booleans(boolean[] values) {
        StringJoiner j = new StringJoiner(", ", "[", "]");
        for (boolean v : values) {
            j.add(Boolean.toString(v));
        }
        return j.toString();
    }

    /** {@code stablehlo.cholesky %a[, lower = true] {..} : T}. */
    private boolean cholesky(Operation op) {
        Attribute lower = attr(op, "lower");
        if (op.numOperands() != 1
                || op.numResults() != 1
                || (lower != null && !(lower instanceof BoolAttr))) {
            return false;
        }
        out.append(op.name()).append(' ').append(name(op.operand(0)));
        if (lower != null) {
            out.append(", lower = ").append(lower);
        }
        out.append(attrDict(op, Set.of("lower"))).append(" : ").append(sameType(op));
        return true;
    }

    /** {@code stablehlo.reduce_precision %x, format = e5m10 {..} : T}. */
    private boolean reducePrecision(Operation op) {
        String e = intOf(attr(op, "exponent_bits"), IntegerType.I32);
        String m = intOf(attr(op, "mantissa_bits"), IntegerType.I32);
        if (op.numOperands() != 1
                || op.numResults() != 1
                || e == null
                || m == null
                || e.startsWith("-")
                || m.startsWith("-")) {
            return false;
        }
        out.append(op.name()).append(' ').append(name(op.operand(0)));
        out.append(", format = e").append(e).append('m').append(m);
        out.append(attrDict(op, Set.of("exponent_bits", "mantissa_bits")));
        out.append(" : ").append(sameType(op));
        return true;
    }

    /** A single-block region whose entry arguments have the given types. */
    private static boolean singleBlock(Region r, List<Type> argTypes) {
        return r.blocks().size() == 1
                && r.block().arguments().stream().map(Value::type).toList().equals(argTypes);
    }

    /**
     * {@code stablehlo.while(%iterArg = %a, ...) : A, ... [attributes {..}] cond {..} do {..}};
     * both regions name their arguments alike.
     */
    private boolean whileOp(Operation op, int depth) {
        List<Type> types = op.operandTypes();
        if (op.regions().size() != 2
                || !op.resultTypes().equals(types)
                || !singleBlock(op.region(0), types)
                || !singleBlock(op.region(1), types)) {
            return false;
        }
        for (int i = 0; i < op.numOperands(); i++) {
            if (!name(op.region(0).block().argument(i))
                    .equals(name(op.region(1).block().argument(i)))) {
                return false; // not reached: MLIR names both regions' arguments alike
            }
        }
        out.append(op.name()).append('(');
        for (int i = 0; i < op.numOperands(); i++) {
            String n = name(op.region(0).block().argument(i));
            out.append(i > 0 ? ", " : "").append(n).append(" = ").append(name(op.operand(i)));
        }
        out.append(')');
        if (op.numOperands() > 0) {
            out.append(" : ").append(types(types));
        }
        String rest = attrDict(op, Set.of());
        if (!rest.isEmpty()) {
            out.append(" attributes").append(rest);
        }
        defaultDialect.push("");
        out.append('\n');
        indent(depth);
        out.append("cond ");
        region(op.region(0), depth, false);
        out.append(" do ");
        region(op.region(1), depth, false);
        defaultDialect.pop();
        return true;
    }

    /**
     * {@code stablehlo.reduce(%x init: %i) applies stablehlo.add across dimensions = [..] : sig},
     * or the same with {@code reducer(%a: T, %b: T) {body}} when the body is not a single
     * commutative op.
     */
    private boolean reduce(Operation op, int depth) {
        int n = op.numOperands() / 2;
        String dims = i64List(attr(op, "dimensions"));
        if (n == 0 || op.numOperands() % 2 != 0 || dims == null || op.regions().size() != 1) {
            return false;
        }
        List<Value> inputs = op.operands().subList(0, n);
        List<Value> inits = op.operands().subList(n, 2 * n);
        Region body = op.region(0);
        if (body.blocks().size() != 1 || body.block().arguments().size() != 2 * n) {
            return false;
        }
        String compact = compactReducer(inputs, inits, body.block());
        out.append(op.name());
        for (int i = 0; i < n; i++) {
            out.append(i > 0 ? ", (" : "(").append(name(inputs.get(i)));
            out.append(" init: ").append(name(inits.get(i))).append(')');
        }
        if (compact != null) {
            out.append(" applies ").append(compact);
        }
        out.append(" across dimensions = ").append(dims);
        out.append(attrDict(op, Set.of("dimensions"))).append(" : ").append(functionalType(op));
        if (compact == null) {
            out.append('\n');
            indent(depth);
            out.append(" reducer");
            Block block = body.block();
            for (int i = 0; i < n; i++) {
                Value a = block.argument(i);
                Value b = block.argument(i + n);
                out.append('(').append(name(a)).append(": ").append(a.type().mlir());
                out.append(", ").append(name(b)).append(": ").append(b.type().mlir());
                out.append(") ");
            }
            out.append(' ');
            defaultDialect.push("");
            region(body, depth, false);
            defaultDialect.pop();
        }
        return true;
    }

    /** The {@code applies} op of a reduce body that the parser rebuilds exactly, else null. */
    private static String compactReducer(List<Value> inputs, List<Value> inits, Block block) {
        if (inputs.size() != 1 || block.operations().size() != 2) {
            return null;
        }
        Operation inner = block.operations().get(0);
        Operation ret = block.operations().get(1);
        Type init = inits.getFirst().type();
        if (!COMPACT_REDUCERS.contains(inner.name())
                || !inner.allAttributes().isEmpty()
                || !inner.regions().isEmpty()
                || inner.numResults() != 1
                || !inner.operands().equals(block.arguments())
                || !inner.result(0).type().equals(init)
                || !block.argument(0).type().equals(init)
                || !block.argument(1).type().equals(init)
                || !(inputs.getFirst().type() instanceof TensorType in)
                || !init.equals(TensorType.of(in.elementType()))
                || !ret.name().equals("stablehlo.return")
                || !ret.allAttributes().isEmpty()
                || ret.numOperands() != 1
                || ret.operand(0) != inner.result(0)) {
            return null;
        }
        return inner.name();
    }
}
