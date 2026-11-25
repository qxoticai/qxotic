package com.qxotic.jota.stablehlo.text;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
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
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Parser for StableHLO programs in MLIR text form: the generic operation form, the builtin types
 * and attributes, and the custom (pretty) assembly of the StableHLO, {@code func} and {@code
 * builtin} operations, as printed by {@code stablehlo-opt} and JAX.
 *
 * <p>Operations of other dialects ({@code chlo.*}, {@code check.*}...) are accepted in generic form
 * and in the common pretty shapes ({@code op %a, %b : type}, {@code op %a : T -> R}).
 *
 * <p>Not supported yet: multi-block CFG successors, {@code {-# dialect_resources #-}} blobs (the
 * section is skipped; {@code dense_resource} attributes stay opaque).
 */
public final class MlirParser extends AsmParser {

    private Scope scope;

    /** Source line of every parsed operation (synthesized ones excluded). */
    private final Map<Operation, Integer> lines = new java.util.IdentityHashMap<>();

    private int[] lineStarts;

    private static final class Scope {
        final Scope parent;
        final Map<String, List<Value>> names = new HashMap<>();

        Scope(Scope parent) {
            this.parent = parent;
        }
    }

    private record ResultGroup(String name, int count) {}

    private record TypeSig(List<Type> inputs, List<Type> results, boolean functional) {}

    private record NamedType(String name, Type type) {}

    private MlirParser(String src, String sourceName) {
        super(src, sourceName);
    }

    // ================================================================== entry points

    public static Program parse(String text) {
        return parse(text, "<input>");
    }

    public static Program parse(String text, String sourceName) {
        return new MlirParser(text, sourceName).parseTopLevel();
    }

    /** A parsed program with the source line of each of its operations. */
    public record Parsed(Program program, Map<Operation, Integer> lines) {

        /** Source line of {@code op}, or -1 when it was not read from the source. */
        public int line(Operation op) {
            return lines.getOrDefault(op, -1);
        }
    }

    /** Parses and records the source line of every operation (for diagnostics). */
    public static Parsed parseWithLines(String text, String sourceName) {
        MlirParser p = new MlirParser(text, sourceName);
        Program program = p.parseTopLevel();
        return new Parsed(program, java.util.Collections.unmodifiableMap(p.lines));
    }

    public static Program parse(Path file) throws IOException {
        return parse(Files.readString(file, StandardCharsets.UTF_8), file.toString());
    }

    public static Type parseType(String text) {
        MlirParser p = new MlirParser(text, "<type>");
        Type t = p.type();
        if (!p.atEnd()) {
            throw p.error("trailing characters after type");
        }
        return t;
    }

    public static Attribute parseAttribute(String text) {
        MlirParser p = new MlirParser(text, "<attribute>");
        Attribute a = p.attribute();
        if (!p.atEnd()) {
            throw p.error("trailing characters after attribute");
        }
        return a;
    }

    /** Splits a lit test file on {@code // -----} separator lines. */
    public static List<String> splitInputFile(String text) {
        List<String> chunks = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        for (String line : text.split("\n", -1)) {
            if (line.trim().equals("// -----")) {
                chunks.add(current.toString());
                current.setLength(0);
                current.append('\n'); // keep line numbers stable-ish
            } else {
                current.append(line).append('\n');
            }
        }
        chunks.add(current.toString());
        return chunks;
    }

    // ================================================================== top level

    private Program parseTopLevel() {
        scope = new Scope(null);
        Block block = new Block();
        while (!atEnd()) {
            char c = peekChar();
            if (c == '#' && isAliasDefinition()) {
                pos++;
                String name = suffixId();
                expect("=");
                if (peekKeyword("loc")) {
                    skipLocation();
                } else {
                    attributeAliases.put(name, attribute());
                }
            } else if (c == '!' && isAliasDefinition()) {
                pos++;
                String name = suffixId();
                expect("=");
                typeAliases.put(name, type());
            } else if (peek("{-#")) {
                int end = src.indexOf("#-}", pos);
                if (end < 0) {
                    throw error("unterminated {-# section");
                }
                pos = end + 3;
            } else {
                block.addOperation(operation());
            }
        }
        List<Operation> ops = block.operations();
        if (ops.size() == 1 && ops.getFirst().name().equals("builtin.module")) {
            return new Program(ops.getFirst());
        }
        return new Program(
                Operation.create(
                        "builtin.module",
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(),
                        List.of(Region.of(block))));
    }

    private boolean isAliasDefinition() {
        int save = pos;
        try {
            pos++;
            suffixId();
            return peek("=");
        } catch (ParseException e) {
            return false;
        } finally {
            pos = save;
        }
    }

    private void skipLocation() {
        if (peekKeyword("loc") && afterKeywordIs("loc", '(')) {
            pos += 3;
            balanced('(', ')');
        }
    }

    private boolean afterKeywordIs(String kw, char c) {
        int save = pos;
        pos += kw.length();
        boolean r = peekChar() == c;
        pos = save;
        return r;
    }

    // ================================================================== values & scopes

    private Value lookup(String name, int index, int at) {
        for (Scope s = scope; s != null; s = s.parent) {
            List<Value> group = s.names.get(name);
            if (group != null) {
                if (index >= group.size()) {
                    throw errorAt(at, "result #" + index + " out of range for %" + name);
                }
                return group.get(index);
            }
        }
        throw errorAt(at, "use of undefined value %" + name);
    }

    private void bind(String name, List<Value> values, int at) {
        if (scope.names.putIfAbsent(name, values) != null) {
            throw errorAt(at, "redefinition of %" + name);
        }
    }

    /** {@code %name} or {@code %name#index}. */
    private Value operandRef() {
        ws();
        int at = pos;
        String name = valueName();
        int index = 0;
        if (charAt(0) == '#') {
            pos++;
            index = (int) integer();
        }
        return lookup(name, index, at);
    }

    private List<Value> operandList() {
        List<Value> values = new ArrayList<>();
        if (peekChar() != '%') {
            return values;
        }
        do {
            values.add(operandRef());
        } while (tryConsume(","));
        return values;
    }

    // ================================================================== operations

    private Operation operation() {
        ws();
        int start = pos;
        List<ResultGroup> groups = new ArrayList<>();
        if (charAt(0) == '%') {
            do {
                String name = valueName();
                int count = 1;
                if (tryConsume(":")) {
                    count = (int) integer();
                }
                groups.add(new ResultGroup(name, count));
            } while (tryConsume(","));
            expect("=");
        }
        int numResults = groups.stream().mapToInt(ResultGroup::count).sum();
        List<String> hints = new ArrayList<>();
        for (ResultGroup g : groups) {
            for (int i = 0; i < g.count; i++) {
                hints.add(g.count == 1 ? g.name : g.name + "#" + i);
            }
        }
        Operation op;
        ws();
        if (charAt(0) == '"') {
            String name = string();
            op = genericTail(name, hints);
        } else {
            op = pretty(canonicalName(ident()), numResults, hints);
        }
        skipLocation();
        if (!groups.isEmpty()) {
            if (op.numResults() != numResults) {
                throw errorAt(
                        start,
                        "operation defines "
                                + numResults
                                + " results but its type has "
                                + op.numResults());
            }
            int r = 0;
            for (ResultGroup g : groups) {
                bind(g.name, op.results().subList(r, r + g.count), start);
                r += g.count;
            }
        }
        lines.put(op, lineOf(start));
        return op;
    }

    /** 1-based line of a source offset. */
    private int lineOf(int offset) {
        if (lineStarts == null) {
            List<Integer> starts = new ArrayList<>();
            starts.add(0);
            for (int i = 0; i < src.length(); i++) {
                if (src.charAt(i) == '\n') {
                    starts.add(i + 1);
                }
            }
            lineStarts = starts.stream().mapToInt(Integer::intValue).toArray();
        }
        int i = java.util.Arrays.binarySearch(lineStarts, offset);
        return i >= 0 ? i + 1 : -i - 1;
    }

    private static String canonicalName(String name) {
        return switch (name) {
            case "return" -> "func.return";
            case "call" -> "func.call";
            case "module" -> "builtin.module";
            default -> name;
        };
    }

    /**
     * Generic form after the quoted name: {@code (operands) <{props}> ({regions}) {attrs} : fn}.
     */
    private Operation genericTail(String name, List<String> hints) {
        expect("(");
        List<Value> operands = new ArrayList<>();
        if (!tryConsume(")")) {
            do {
                operands.add(operandRef());
            } while (tryConsume(","));
            expect(")");
        }
        if (peek("[")) {
            throw error("block successors are not supported");
        }
        List<NamedAttribute> props = List.of();
        if (peek("<{")) {
            expect("<");
            props = attrDict();
            expect(">");
        }
        List<Region> regions = new ArrayList<>();
        if (peek("(")) {
            int save = pos;
            pos++;
            if (peekChar() == '{') {
                do {
                    regions.add(region(null));
                } while (tryConsume(","));
                expect(")");
            } else {
                pos = save;
            }
        }
        List<NamedAttribute> attrs = peek("{") ? attrDict() : List.of();
        expect(":");
        int typeAt = pos;
        Type t = type();
        if (!(t instanceof FunctionType ft)) {
            throw errorAt(typeAt, "expected a function type in generic operation");
        }
        checkOperandTypes(operands, ft.inputs(), typeAt);
        return Operation.create(name, operands, ft.results(), hints, props, attrs, regions);
    }

    // ================================================================== regions

    /** Parses {@code { [^bb0(args):] ops ... }}; {@code entryArgs} pre-declares entry arguments. */
    private Region region(List<NamedType> entryArgs) {
        expect("{");
        Scope outer = scope;
        scope = new Scope(outer);
        Region region = new Region();
        Block block = new Block();
        if (entryArgs != null) {
            for (NamedType a : entryArgs) {
                bind(a.name, List.of(block.addArgument(a.type, a.name)), pos);
            }
        }
        boolean blockStarted = entryArgs != null;
        while (true) {
            ws();
            if (charAt(0) == '}') {
                break;
            }
            if (charAt(0) == '^') {
                if (blockStarted || !block.operations().isEmpty()) {
                    if (entryArgs != null && region.isEmpty() && block.operations().isEmpty()) {
                        throw error("block label after pre-declared region arguments");
                    }
                    region.addBlock(block);
                    block = new Block();
                }
                pos++;
                block.setLabel(suffixId());
                if (tryConsume("(")) {
                    if (!tryConsume(")")) {
                        do {
                            int at = pos;
                            String n = valueName();
                            expect(":");
                            Type t = type();
                            skipLocation();
                            bind(n, List.of(block.addArgument(t, n)), at);
                        } while (tryConsume(","));
                        expect(")");
                    }
                }
                expect(":");
                blockStarted = true;
                continue;
            }
            block.addOperation(operation());
            blockStarted = true;
        }
        expect("}");
        if (blockStarted) {
            region.addBlock(block);
        }
        scope = outer;
        return region;
    }

    // ================================================================== pretty forms

    private Operation pretty(String name, int numResults, List<String> hints) {
        return switch (name) {
            case "builtin.module" -> moduleOp(hints);
            case "func.func" -> funcOp(hints);
            case "func.call" -> callOp(hints);
            case "func.return", "stablehlo.return" -> returnOp(name, hints);
            case "stablehlo.constant" -> constantOp(numResults, hints);
            case "stablehlo.compare" -> compareOp(numResults, hints);
            case "stablehlo.slice" -> sliceOp(numResults, hints);
            case "stablehlo.get_tuple_element" -> getTupleElementOp(numResults, hints);
            case "stablehlo.while" -> whileOp(hints);
            case "stablehlo.reduce" -> reduceOp(numResults, hints);
            case "stablehlo.convolution" -> convolutionOp(numResults, hints);
            case "stablehlo.custom_call" -> customCallOp(numResults, hints);
            case "stablehlo.composite" -> compositeOp(numResults, hints);
            case "stablehlo.dot_general", "stablehlo.dot" -> dotOp(name, numResults, hints);
            case "stablehlo.optimization_barrier" -> optimizationBarrierOp(name, numResults, hints);
            case "chlo.top_k" -> topKOp(name, numResults, hints);
            case "arith.constant" -> arithConstantOp(hints);
            default -> {
                if (name.startsWith("check.") && name.endsWith("_const")) {
                    yield checkConstOp(name, hints);
                }
                yield heuristicOp(name, numResults, hints);
            }
        };
    }

    /**
     * The commutative binary ops of the stablehlo dialect ({@code applies} of a compact reduce).
     */
    private static final java.util.Set<String> COMPACT_REDUCERS =
            java.util.Set.of(
                    "stablehlo.add",
                    "stablehlo.multiply",
                    "stablehlo.maximum",
                    "stablehlo.minimum",
                    "stablehlo.and",
                    "stablehlo.or",
                    "stablehlo.xor");

    private Operation build(
            String name,
            List<Value> operands,
            List<Type> resultTypes,
            List<String> hints,
            List<NamedAttribute> props,
            List<NamedAttribute> attrs,
            List<Region> regions) {
        return Operation.create(name, operands, resultTypes, hints, props, attrs, regions);
    }

    private List<NamedAttribute> optionalAttrDict() {
        return peek("{") ? attrDict() : List.of();
    }

    private List<NamedAttribute> optionalAttributesKeyword() {
        return tryKeyword("attributes") ? attrDict() : List.of();
    }

    /** {@code module [@name] [attributes {...}] { ... }}. */
    private Operation moduleOp(List<String> hints) {
        List<NamedAttribute> props = new ArrayList<>();
        if (peek("@")) {
            props.add(new NamedAttribute("sym_name", new StringAttr(symbolName())));
        }
        List<NamedAttribute> attrs = optionalAttributesKeyword();
        Region body = region(List.of());
        return build("builtin.module", List.of(), List.of(), hints, props, attrs, List.of(body));
    }

    /**
     * {@code func.func [visibility] @name(%a: T {attrs}, ...) [-> (R {attrs}, ...)] [attributes
     * {...}] [{ body }]}.
     */
    private Operation funcOp(List<String> hints) {
        String visibility = null;
        for (String v : new String[] {"public", "private", "nested"}) {
            if (tryKeyword(v)) {
                visibility = v;
            }
        }
        String symbol = symbolName();
        expect("(");
        List<NamedType> args = new ArrayList<>();
        List<Attribute> argAttrs = new ArrayList<>();
        boolean anyArgAttr = false;
        if (!tryConsume(")")) {
            do {
                String n = null;
                if (peekChar() == '%') { // declarations may list bare types
                    n = valueName();
                    expect(":");
                }
                Type t = type();
                DictionaryAttr d =
                        peek("{") ? new DictionaryAttr(attrDict()) : DictionaryAttr.EMPTY;
                anyArgAttr |= !d.isEmpty();
                skipLocation();
                args.add(new NamedType(n, t));
                argAttrs.add(d);
            } while (tryConsume(","));
            expect(")");
        }
        List<Type> results = new ArrayList<>();
        List<Attribute> resAttrs = new ArrayList<>();
        boolean anyResAttr = false;
        if (tryConsume("->")) {
            ws();
            boolean parens = charAt(0) == '(' && !isFunctionTypeAhead();
            if (parens) {
                pos++;
                if (!tryConsume(")")) {
                    do {
                        results.add(type());
                        DictionaryAttr d =
                                peek("{") ? new DictionaryAttr(attrDict()) : DictionaryAttr.EMPTY;
                        anyResAttr |= !d.isEmpty();
                        resAttrs.add(d);
                    } while (tryConsume(","));
                    expect(")");
                }
            } else {
                results.add(type());
                DictionaryAttr d =
                        peek("{") && isAttrDictAhead()
                                ? new DictionaryAttr(attrDict())
                                : DictionaryAttr.EMPTY;
                anyResAttr |= !d.isEmpty();
                resAttrs.add(d);
            }
        }
        List<NamedAttribute> attrs = optionalAttributesKeyword();
        Region body = peek("{") ? region(args) : new Region();
        List<NamedAttribute> props = new ArrayList<>();
        props.add(new NamedAttribute("sym_name", new StringAttr(symbol)));
        props.add(
                new NamedAttribute(
                        "function_type",
                        new TypeAttr(
                                new FunctionType(
                                        args.stream().map(NamedType::type).toList(), results))));
        if (visibility != null) {
            props.add(new NamedAttribute("sym_visibility", new StringAttr(visibility)));
        }
        if (anyArgAttr) {
            props.add(new NamedAttribute("arg_attrs", new ArrayAttr(argAttrs)));
        }
        if (anyResAttr) {
            props.add(new NamedAttribute("res_attrs", new ArrayAttr(resAttrs)));
        }
        return build("func.func", List.of(), List.of(), hints, props, attrs, List.of(body));
    }

    /** After a single result type, '{' starts a result attr dict only if a key follows. */
    private boolean isAttrDictAhead() {
        int save = pos;
        try {
            expect("{");
            ws();
            if (charAt(0) == '}') {
                return true;
            }
            if (charAt(0) == '"') {
                string();
            } else {
                ident();
            }
            ws();
            return charAt(0) == '=' || charAt(0) == ',' || charAt(0) == '}';
        } catch (ParseException e) {
            return false;
        } finally {
            pos = save;
        }
    }

    /** True when the '(' at the cursor starts a function type {@code (A) -> B}. */
    private boolean isFunctionTypeAhead() {
        int save = pos;
        try {
            balanced('(', ')');
            return peek("->");
        } catch (ParseException e) {
            return false;
        } finally {
            pos = save;
        }
    }

    /** {@code call @f(%a, ...) [attrs] : (A) -> R}. */
    private Operation callOp(List<String> hints) {
        String callee = symbolName();
        expect("(");
        List<Value> operands = operandList();
        expect(")");
        List<NamedAttribute> attrs = optionalAttrDict();
        expect(":");
        int at = pos;
        Type t = type();
        if (!(t instanceof FunctionType ft)) {
            throw errorAt(at, "expected function type");
        }
        checkOperandTypes(operands, ft.inputs(), at);
        List<NamedAttribute> props =
                List.of(new NamedAttribute("callee", SymbolRefAttr.of(callee)));
        return build("func.call", operands, ft.results(), hints, props, attrs, List.of());
    }

    /** {@code return [%a, %b : A, B]}. */
    private Operation returnOp(String name, List<String> hints) {
        List<Value> operands = operandList();
        List<NamedAttribute> attrs = optionalAttrDict();
        if (!operands.isEmpty()) {
            expect(":");
            int at = pos;
            List<Type> types = new ArrayList<>();
            do {
                types.add(type());
            } while (tryConsume(","));
            checkOperandTypes(operands, types, at);
        }
        return build(name, operands, List.of(), hints, List.of(), attrs, List.of());
    }

    /** {@code stablehlo.constant [attrs] dense<...> : T}, or the generic-like {@code ()} form. */
    private Operation constantOp(int numResults, List<String> hints) {
        if (peek("(")) {
            return genericTail("stablehlo.constant", hints);
        }
        List<NamedAttribute> attrs = optionalAttrDict();
        int at = pos;
        Attribute value = attribute();
        if (!(value instanceof DenseElementsAttr dense)) {
            throw errorAt(at, "stablehlo.constant value must be a dense elements attribute");
        }
        Type resultType = dense.type();
        if (tryConsume(":")) { // quantized result type spelled after the value
            resultType = type();
        }
        return build(
                "stablehlo.constant",
                List.of(),
                List.of(resultType),
                hints,
                List.of(new NamedAttribute("value", value)),
                attrs,
                List.of());
    }

    /** {@code stablehlo.compare DIR, %a, %b[, TYPE] [attrs] : sig}. */
    private Operation compareOp(int numResults, List<String> hints) {
        List<NamedAttribute> props = new ArrayList<>();
        props.add(
                new NamedAttribute(
                        "comparison_direction",
                        EnumAttr.stablehlo("comparison_direction", ident())));
        expect(",");
        Value lhs = operandRef();
        expect(",");
        Value rhs = operandRef();
        if (tryConsume(",")) {
            props.add(
                    new NamedAttribute(
                            "compare_type", EnumAttr.stablehlo("comparison_type", ident())));
        }
        return finishPretty(
                "stablehlo.compare", List.of(lhs, rhs), numResults, hints, props, List.of());
    }

    /** {@code stablehlo.slice %x [s:l(:st)?, ...] [attrs] : sig}. */
    private Operation sliceOp(int numResults, List<String> hints) {
        Value operand = operandRef();
        expect("[");
        List<Long> starts = new ArrayList<>();
        List<Long> limits = new ArrayList<>();
        List<Long> strides = new ArrayList<>();
        if (!tryConsume("]")) {
            do {
                starts.add(integer());
                expect(":");
                limits.add(integer());
                strides.add(tryConsume(":") ? integer() : 1L);
            } while (tryConsume(","));
            expect("]");
        }
        List<NamedAttribute> props =
                List.of(
                        new NamedAttribute("start_indices", i64Array(starts)),
                        new NamedAttribute("limit_indices", i64Array(limits)),
                        new NamedAttribute("strides", i64Array(strides)));
        return finishPretty(
                "stablehlo.slice", List.of(operand), numResults, hints, props, List.of());
    }

    /** {@code stablehlo.get_tuple_element %t[i] [attrs] : sig}. */
    private Operation getTupleElementOp(int numResults, List<String> hints) {
        Value operand = operandRef();
        expect("[");
        long index = integer();
        expect("]");
        List<NamedAttribute> props = List.of(new NamedAttribute("index", IntegerAttr.i32(index)));
        return finishPretty(
                "stablehlo.get_tuple_element",
                List.of(operand),
                numResults,
                hints,
                props,
                List.of());
    }

    /** {@code stablehlo.while(%it = %init, ...) : T, ... [attributes {..}] cond {..} do {..}}. */
    private Operation whileOp(List<String> hints) {
        expect("(");
        List<String> names = new ArrayList<>();
        List<Value> inits = new ArrayList<>();
        if (!tryConsume(")")) {
            do {
                names.add(valueName());
                expect("=");
                inits.add(operandRef());
            } while (tryConsume(","));
            expect(")");
        }
        List<Type> types = new ArrayList<>();
        if (tryConsume(":")) {
            do {
                types.add(type());
            } while (tryConsume(","));
        }
        if (types.size() != inits.size()) {
            throw error("while: " + inits.size() + " operands but " + types.size() + " types");
        }
        List<NamedAttribute> attrs = optionalAttributesKeyword();
        List<NamedType> args = new ArrayList<>();
        for (int i = 0; i < names.size(); i++) {
            args.add(new NamedType(names.get(i), types.get(i)));
        }
        expectKeyword("cond");
        Region cond = region(args);
        expectKeyword("do");
        Region body = region(args);
        return build("stablehlo.while", inits, types, hints, List.of(), attrs, List.of(cond, body));
    }

    /**
     * {@code stablehlo.reduce(%x init: %i), ... applies OP across dimensions = [..] : sig} or
     * {@code ... across dimensions = [..] : sig reducer(%a: T, %b: T) ... { body }}.
     */
    private Operation reduceOp(int numResults, List<String> hints) {
        List<Value> inputs = new ArrayList<>();
        List<Value> inits = new ArrayList<>();
        do {
            expect("(");
            inputs.add(operandRef());
            expectKeyword("init");
            expect(":");
            inits.add(operandRef());
            expect(")");
        } while (tryConsume(","));
        String applies = null;
        if (tryKeyword("applies")) {
            int at = pos;
            applies = ident();
            if (!COMPACT_REDUCERS.contains(applies)) {
                throw errorAt(
                        at,
                        "expected the inner-op to be a commutative binary-op from the stablehlo"
                                + " dialect, with zero region, producing single result");
            }
        }
        expectKeyword("across");
        expectKeyword("dimensions");
        expect("=");
        DenseArrayAttr dims = i64List();
        List<NamedAttribute> attrs = optionalAttrDict();
        List<Value> operands = new ArrayList<>(inputs);
        operands.addAll(inits);
        expect(":");
        int at = pos;
        List<Type> resultTypes =
                resolveTypes("stablehlo.reduce", operands, numResults, typeSig(), at);
        Region body;
        if (applies != null) {
            Block block = new Block();
            List<Value> lhs = new ArrayList<>();
            List<Value> rhs = new ArrayList<>();
            for (Value init : inits) {
                lhs.add(block.addArgument(init.type(), null));
            }
            for (Value init : inits) {
                rhs.add(block.addArgument(init.type(), null));
            }
            List<Value> args = new ArrayList<>(lhs);
            args.addAll(rhs);
            Operation combiner =
                    Operation.create(
                            applies,
                            args,
                            inits.stream().map(Value::type).toList(),
                            List.of(),
                            List.of(),
                            List.of());
            block.addOperation(combiner);
            block.addOperation(
                    Operation.create(
                            "stablehlo.return",
                            combiner.results(),
                            List.of(),
                            List.of(),
                            List.of(),
                            List.of()));
            body = Region.of(block);
        } else {
            expectKeyword("reducer");
            List<NamedType> lhs = new ArrayList<>();
            List<NamedType> rhs = new ArrayList<>();
            for (int i = 0; i < inputs.size(); i++) {
                expect("(");
                String a = valueName();
                expect(":");
                Type ta = type();
                expect(",");
                String b = valueName();
                expect(":");
                Type tb = type();
                expect(")");
                lhs.add(new NamedType(a, ta));
                rhs.add(new NamedType(b, tb));
            }
            List<NamedType> args = new ArrayList<>(lhs);
            args.addAll(rhs);
            body = region(args);
        }
        return build(
                "stablehlo.reduce",
                operands,
                resultTypes,
                hints,
                List.of(new NamedAttribute("dimensions", dims)),
                attrs,
                List.of(body));
    }

    /**
     * {@code stablehlo.convolution(%l, %r) dim_numbers = [b, 0, f]x[0, i, o]->[b, 0, f], window =
     * {stride = [..], pad = [[..]], lhs_dilate = [..], rhs_dilate = [..], reverse = [..]} {attrs} :
     * sig}.
     */
    private Operation convolutionOp(int numResults, List<String> hints) {
        expect("(");
        Value lhs = operandRef();
        expect(",");
        Value rhs = operandRef();
        expect(")");
        expectKeyword("dim_numbers");
        expect("=");
        String layout =
                balanced('[', ']')
                        + (consumeOrFail("x") ? "x" : "")
                        + balanced('[', ']')
                        + (consumeOrFail("->") ? "->" : "")
                        + balanced('[', ']');
        List<NamedAttribute> props = new ArrayList<>();
        props.add(
                new NamedAttribute(
                        "dimension_numbers",
                        ConvDimensionNumbers.parseCompact(layout).toAttribute()));
        expect(",");
        expectKeyword("window");
        expect("=");
        expect("{");
        if (!tryConsume("}")) {
            do {
                String key = ident();
                expect("=");
                switch (key) {
                    case "stride" -> props.add(new NamedAttribute("window_strides", i64List()));
                    case "lhs_dilate" -> props.add(new NamedAttribute("lhs_dilation", i64List()));
                    case "rhs_dilate" -> props.add(new NamedAttribute("rhs_dilation", i64List()));
                    case "pad" -> props.add(new NamedAttribute("padding", paddingPairs()));
                    case "reverse" -> props.add(new NamedAttribute("window_reversal", boolList()));
                    default -> throw error("unknown convolution window attribute '" + key + "'");
                }
            } while (tryConsume(","));
            expect("}");
        }
        return finishPretty(
                "stablehlo.convolution", List.of(lhs, rhs), numResults, hints, props, List.of());
    }

    private boolean consumeOrFail(String s) {
        expect(s);
        return true;
    }

    /** {@code [[lo, hi], ...]} as {@code dense<...> : tensor<Nx2xi64>}. */
    private DenseElementsAttr paddingPairs() {
        expect("[");
        List<Long> values = new ArrayList<>();
        if (!tryConsume("]")) {
            do {
                expect("[");
                values.add(integer());
                expect(",");
                values.add(integer());
                expect("]");
            } while (tryConsume(","));
            expect("]");
        }
        TensorType tt = TensorType.of(IntegerType.I64, values.size() / 2, 2);
        byte[] data = new byte[values.size() * 8];
        for (int i = 0; i < values.size(); i++) {
            writeLE(data, i * 8, 8, values.get(i));
        }
        return new DenseElementsAttr(tt, data, false);
    }

    private DenseArrayAttr boolList() {
        expect("[");
        List<Boolean> values = new ArrayList<>();
        if (!tryConsume("]")) {
            do {
                if (tryKeyword("true")) {
                    values.add(true);
                } else if (tryKeyword("false")) {
                    values.add(false);
                } else {
                    values.add(integer() != 0);
                }
            } while (tryConsume(","));
            expect("]");
        }
        boolean[] b = new boolean[values.size()];
        for (int i = 0; i < b.length; i++) {
            b[i] = values.get(i);
        }
        return DenseArrayAttr.i1(b);
    }

    /** {@code stablehlo.custom_call @target(%a, ...) [attrs] : sig}. */
    private Operation customCallOp(int numResults, List<String> hints) {
        String target = symbolName();
        expect("(");
        List<Value> operands = operandList();
        expect(")");
        List<NamedAttribute> props =
                List.of(new NamedAttribute("call_target_name", new StringAttr(target)));
        return finishPretty("stablehlo.custom_call", operands, numResults, hints, props, List.of());
    }

    /** {@code stablehlo.composite "name" %a, ... [attrs] : sig}. */
    private Operation compositeOp(int numResults, List<String> hints) {
        String compositeName = string();
        List<Value> operands = operandList();
        List<NamedAttribute> props =
                List.of(new NamedAttribute("name", new StringAttr(compositeName)));
        List<Region> regions = new ArrayList<>();
        if (peek("(")) { // composite_regions: ({ ... }, { ... })
            pos++;
            do {
                regions.add(region(null));
            } while (tryConsume(","));
            expect(")");
        }
        return finishPretty("stablehlo.composite", operands, numResults, hints, props, regions);
    }

    /**
     * {@code stablehlo.dot_general %a, %b, [batching_dims = [..] x [..],] contracting_dims = [..] x
     * [..][, precision = [..]][, algorithm = <..>] [attrs] : sig}; {@code stablehlo.dot %a, %b[,
     * precision = [..]]}.
     */
    private Operation dotOp(String name, int numResults, List<String> hints) {
        List<Value> operands = new ArrayList<>();
        operands.add(operandRef());
        expect(",");
        operands.add(operandRef());
        List<NamedAttribute> props = new ArrayList<>();
        if (name.equals("stablehlo.dot_general")) {
            expect(",");
            if (peek("[")) {
                pos++;
                operands.addAll(operandList());
                expect("]");
                expect(",");
            }
            long[] lb = new long[0];
            long[] rb = new long[0];
            if (tryKeyword("batching_dims")) {
                expect("=");
                lb = i64List().values();
                expectKeyword("x");
                rb = i64List().values();
                expect(",");
            }
            expectKeyword("contracting_dims");
            expect("=");
            long[] lc = i64List().values();
            expectKeyword("x");
            long[] rc = i64List().values();
            props.add(
                    new NamedAttribute(
                            "dot_dimension_numbers",
                            new DotDimensionNumbers(lb, rb, lc, rc).toAttribute()));
        }
        while (tryConsume(",")) {
            String key = ident();
            expect("=");
            switch (key) {
                case "precision" -> {
                    expect("[");
                    List<Attribute> values = new ArrayList<>();
                    if (!tryConsume("]")) {
                        do {
                            values.add(EnumAttr.stablehlo("precision", ident()));
                        } while (tryConsume(","));
                        expect("]");
                    }
                    props.add(new NamedAttribute("precision_config", new ArrayAttr(values)));
                }
                case "algorithm" ->
                        props.add(
                                new NamedAttribute(
                                        "algorithm",
                                        dialectAttrBody("stablehlo", "dot_algorithm")));
                default -> throw error("unknown dot attribute '" + key + "'");
            }
        }
        return finishPretty(name, operands, numResults, hints, props, List.of());
    }

    /** {@code stablehlo.optimization_barrier %a, %b : A, B} or {@code ()}. */
    private Operation optimizationBarrierOp(String name, int numResults, List<String> hints) {
        List<NamedAttribute> attrs = optionalAttrDict();
        if (tryConsume("(")) {
            expect(")");
            return build(name, List.of(), List.of(), hints, List.of(), attrs, List.of());
        }
        List<Value> operands = operandList();
        expect(":");
        int at = pos;
        List<Type> types = new ArrayList<>();
        do {
            types.add(type());
        } while (tryConsume(","));
        checkOperandTypes(operands, types, at);
        return build(name, operands, types, hints, List.of(), attrs, List.of());
    }

    /** {@code chlo.top_k(%x, k = 2) : T -> (T, I)}. */
    private Operation topKOp(String name, int numResults, List<String> hints) {
        expect("(");
        Value operand = operandRef();
        expect(",");
        expectKeyword("k");
        expect("=");
        long k = integer();
        expect(")");
        return finishPretty(
                name,
                List.of(operand),
                numResults,
                hints,
                List.of(new NamedAttribute("k", IntegerAttr.i64(k))),
                List.of());
    }

    /** {@code arith.constant [attrs] 0 : index} / {@code arith.constant dense<..> : T}. */
    private Operation arithConstantOp(List<String> hints) {
        List<NamedAttribute> attrs = optionalAttrDict();
        int at = pos;
        Attribute value = attribute();
        Type type =
                switch (value) {
                    case DenseElementsAttr d -> d.type();
                    case IntegerAttr i -> i.type();
                    case com.qxotic.jota.stablehlo.attr.FloatAttr f -> f.type();
                    case BoolAttr b -> IntegerType.I1;
                    default -> throw errorAt(at, "unsupported arith.constant value " + value);
                };
        return build(
                "arith.constant",
                List.of(),
                List.of(type),
                hints,
                List.of(new NamedAttribute("value", value)),
                attrs,
                List.of());
    }

    /** {@code check.expect_*_const %x, dense<...> : T}. */
    private Operation checkConstOp(String name, List<String> hints) {
        Value operand = operandRef();
        expect(",");
        Attribute value = attribute();
        List<NamedAttribute> attrs = optionalAttrDict();
        return build(
                name,
                List.of(operand),
                List.of(),
                hints,
                List.of(new NamedAttribute("value", value)),
                attrs,
                List.of());
    }

    /**
     * The common pretty shape: {@code op %a, %b[, kw = value]* [attrs] : sig}, where {@code sig} is
     * a single type ({@code SameOperandsAndResultType}), a type list, or a function type.
     */
    private Operation heuristicOp(String name, int numResults, List<String> hints) {
        List<Value> operands = new ArrayList<>();
        List<NamedAttribute> props = new ArrayList<>();
        while (true) {
            ws();
            if (charAt(0) == '%') {
                operands.add(operandRef());
            } else if (peekIdent() && isKeywordAssignment()) {
                props.addAll(keywordAttribute(name));
            } else {
                break;
            }
            if (!tryConsume(",")) {
                break;
            }
        }
        return finishPretty(name, operands, numResults, hints, props, List.of());
    }

    /** Parses {@code [attrs] : sig} and builds the operation. */
    private Operation finishPretty(
            String name,
            List<Value> operands,
            int numResults,
            List<String> hints,
            List<NamedAttribute> props,
            List<Region> regions) {
        List<NamedAttribute> attrs = optionalAttrDict();
        expect(":");
        int at = pos;
        List<Type> resultTypes = resolveTypes(name, operands, numResults, typeSig(), at);
        return build(name, operands, resultTypes, hints, props, attrs, regions);
    }

    private boolean isKeywordAssignment() {
        int save = pos;
        try {
            ident();
            return peek("=") && !peek("==");
        } finally {
            pos = save;
        }
    }

    /** Keyword attributes of pretty forms ({@code dims = [..]}, {@code dim = 0}...). */
    private List<NamedAttribute> keywordAttribute(String opName) {
        String kw = ident();
        expect("=");
        String op = opName.substring(opName.indexOf('.') + 1);
        return switch (kw) {
            case "dims" ->
                    List.of(
                            new NamedAttribute(
                                    switch (op) {
                                        case "broadcast_in_dim", "dynamic_broadcast_in_dim" ->
                                                "broadcast_dimensions";
                                        case "transpose" -> "permutation";
                                        default -> "dimensions";
                                    },
                                    i64List()));
            case "dim" ->
                    List.of(
                            new NamedAttribute(
                                    op.equals("iota") || op.equals("dynamic_iota")
                                            ? "iota_dimension"
                                            : "dimension",
                                    IntegerAttr.i64(integer())));
            case "sizes" ->
                    List.of(
                            new NamedAttribute(
                                    op.equals("broadcast") ? "broadcast_sizes" : "slice_sizes",
                                    i64List()));
            case "low" -> List.of(new NamedAttribute("edge_padding_low", i64List()));
            case "high" -> List.of(new NamedAttribute("edge_padding_high", i64List()));
            case "interior" -> List.of(new NamedAttribute("interior_padding", i64List()));
            case "type" ->
                    List.of(
                            new NamedAttribute(
                                    "fft_type", EnumAttr.stablehlo("fft_type", ident())));
            case "length" -> List.of(new NamedAttribute("fft_length", i64List()));
            case "algorithm" ->
                    List.of(
                            new NamedAttribute(
                                    "rng_algorithm", EnumAttr.stablehlo("rng_algorithm", ident())));
            case "distribution" ->
                    List.of(
                            new NamedAttribute(
                                    "rng_distribution",
                                    EnumAttr.stablehlo("rng_distribution", ident())));
            case "lower" -> List.of(new NamedAttribute("lower", BoolAttr.of(booleanKeyword())));
            case "config" -> List.of(new NamedAttribute("einsum_config", new StringAttr(string())));
            case "format" -> {
                int at = pos;
                String f = ident();
                var m = java.util.regex.Pattern.compile("e(\\d+)m(\\d+)").matcher(f);
                if (!m.matches()) {
                    throw errorAt(at, "expected exponent/mantissa format eXmY, got " + f);
                }
                yield List.of(
                        new NamedAttribute(
                                "exponent_bits", IntegerAttr.i32(Long.parseLong(m.group(1)))),
                        new NamedAttribute(
                                "mantissa_bits", IntegerAttr.i32(Long.parseLong(m.group(2)))));
            }
            default -> List.of(new NamedAttribute(kw, keywordValue()));
        };
    }

    /**
     * Value of an unknown pretty keyword: integer lists, untyped numbers (the op's {@code : type}
     * follows, so a number must not swallow it), or any attribute.
     */
    private Attribute keywordValue() {
        ws();
        if (charAt(0) == '[') {
            int save = pos;
            try {
                return i64List();
            } catch (ParseException e) {
                pos = save;
            }
        }
        if (peekNumber()) {
            String tok = numberToken();
            if (isFloatToken(tok)) {
                return com.qxotic.jota.stablehlo.attr.FloatAttr.of(
                        Double.parseDouble(tok), com.qxotic.jota.stablehlo.type.FloatType.F64);
            }
            return IntegerAttr.i64(parseInteger(tok).longValue());
        }
        return attribute();
    }

    private boolean booleanKeyword() {
        if (tryKeyword("true")) {
            return true;
        }
        expectKeyword("false");
        return false;
    }

    private DenseArrayAttr i64List() {
        expect("[");
        List<Long> values = new ArrayList<>();
        if (!tryConsume("]")) {
            do {
                values.add(integer());
            } while (tryConsume(","));
            expect("]");
        }
        return i64Array(values);
    }

    private static DenseArrayAttr i64Array(List<Long> values) {
        return DenseArrayAttr.i64(values.stream().mapToLong(Long::longValue).toArray());
    }

    // ================================================================== type signatures

    /** {@code (A, B) -> R}, {@code A, B -> R}, or a plain type list {@code A[, B...]}. */
    private TypeSig typeSig() {
        ws();
        if (charAt(0) == '(') {
            Type t = type();
            if (t instanceof FunctionType ft) {
                return new TypeSig(ft.inputs(), ft.results(), true);
            }
            throw error("expected function type");
        }
        List<Type> list = new ArrayList<>();
        do {
            list.add(type());
        } while (tryConsume(","));
        if (tryConsume("->")) {
            return new TypeSig(list, resultTypes(), true);
        }
        if (tryKeyword("to")) { // cast-like ops: %x : A to B
            return new TypeSig(list, List.of(type()), true);
        }
        return new TypeSig(list, null, false);
    }

    /** Derives result types (and checks operand types) from a pretty type signature. */
    private List<Type> resolveTypes(
            String name, List<Value> operands, int numResults, TypeSig sig, int at) {
        if (sig.functional) {
            checkOperandTypes(operands, sig.inputs, at);
            if (numResults != 0 && sig.results.size() != numResults) {
                throw errorAt(
                        at,
                        name
                                + ": "
                                + numResults
                                + " results named, "
                                + sig.results.size()
                                + " typed");
            }
            return sig.results;
        }
        List<Type> list = sig.inputs;
        if (name.equals("stablehlo.select") && list.size() == 2 && operands.size() == 3) {
            checkOperandTypes(operands, List.of(list.get(0), list.get(1), list.get(1)), at);
            return Collections.nCopies(numResults, list.get(1));
        }
        if (list.size() == 1) {
            Type t = list.getFirst();
            switch (name) {
                case "stablehlo.tuple" -> {
                    if (!(t instanceof TupleType tt)) {
                        throw errorAt(at, "tuple result type expected");
                    }
                    checkOperandTypes(operands, tt.types(), at);
                    return List.of(t);
                }
                case "stablehlo.complex" -> {
                    if (t instanceof TensorType tt && tt.elementType() instanceof ComplexType ct) {
                        Type part = tt.withElementType(ct.elementType());
                        checkOperandTypes(operands, Collections.nCopies(operands.size(), part), at);
                    }
                    return List.of(t);
                }
                default -> {
                    checkOperandTypes(operands, Collections.nCopies(operands.size(), t), at);
                    return Collections.nCopies(numResults, t);
                }
            }
        }
        if (list.size() == operands.size() && (numResults == list.size() || numResults == 0)) {
            checkOperandTypes(operands, list, at);
            return numResults == 0 ? List.of() : list;
        }
        if (operands.isEmpty() && list.size() == numResults) {
            return list;
        }
        throw errorAt(
                at,
                name
                        + ": cannot match type list "
                        + list
                        + " with "
                        + operands.size()
                        + " operands and "
                        + numResults
                        + " results");
    }

    private void checkOperandTypes(List<Value> operands, List<Type> types, int at) {
        if (operands.size() != types.size()) {
            throw errorAt(
                    at,
                    "expected "
                            + types.size()
                            + " operand types, got "
                            + operands.size()
                            + " operands");
        }
        for (int i = 0; i < types.size(); i++) {
            if (!operands.get(i).type().equals(types.get(i))) {
                throw errorAt(
                        at,
                        "operand #"
                                + i
                                + " has type "
                                + operands.get(i).type()
                                + " but the signature says "
                                + types.get(i));
            }
        }
    }
}
