package com.qxotic.jota.stablehlo.text;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.OpaqueAttr;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.attr.UnitAttr;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.FutureType;
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import com.qxotic.jota.stablehlo.type.QuantizedType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.regex.Pattern;

/** Parses MLIR builtin / StableHLO types and attributes. */
class AsmParser extends AsmLexer {

    private static final Pattern INT_TYPE = Pattern.compile("(i|si|ui)([0-9]+)");

    /** {@code #name = attr} and {@code !name = type} aliases defined at the top of a file. */
    protected final Map<String, Attribute> attributeAliases = new HashMap<>();

    protected final Map<String, Type> typeAliases = new HashMap<>();

    AsmParser(String src, String sourceName) {
        super(src, sourceName);
    }

    // ================================================================== types

    boolean peekType() {
        ws();
        char c = charAt(0);
        if (c == '(' || c == '!') {
            return true;
        }
        if (!isIdentStart(c)) {
            return false;
        }
        int save = pos;
        String id = ident();
        pos = save;
        return isTypeKeyword(id);
    }

    static boolean isTypeKeyword(String id) {
        return id.equals("tensor")
                || id.equals("tuple")
                || id.equals("complex")
                || id.equals("index")
                || id.equals("none")
                || id.equals("memref")
                || id.equals("vector")
                || INT_TYPE.matcher(id).matches()
                || FloatFormat.byName(id) != null;
    }

    Type type() {
        ws();
        char c = charAt(0);
        if (c == '(') {
            return functionType();
        }
        if (c == '!') {
            return dialectType();
        }
        int start = pos;
        String id = typeIdent();
        switch (id) {
            case "tensor" -> {
                int save = pos;
                expect("<");
                boolean unranked = peek("*");
                pos = save;
                if (unranked) { // tensor<*xf32>: not a StableHLO type, kept opaque
                    return new OpaqueType("tensor" + balanced('<', '>'));
                }
                return tensorType();
            }
            case "tuple" -> {
                expect("<");
                List<Type> types = new ArrayList<>();
                if (!tryConsume(">")) {
                    do {
                        types.add(type());
                    } while (tryConsume(","));
                    expect(">");
                }
                return new TupleType(types);
            }
            case "complex" -> {
                expect("<");
                Type inner = type();
                expect(">");
                if (!(inner instanceof FloatType ft)) {
                    throw errorAt(start, "complex element type must be a float type");
                }
                return new ComplexType(ft);
            }
            case "index" -> {
                return IndexType.INSTANCE;
            }
            case "none" -> {
                return new OpaqueType("none");
            }
            case "memref", "vector" -> {
                return new OpaqueType(id + balanced('<', '>'));
            }
            default -> {
                ElementType et = scalarType(id);
                if (et == null) {
                    throw errorAt(start, "unknown type '" + id + "'");
                }
                return et;
            }
        }
    }

    /** Type keywords: letters and digits only (no '.', which would swallow {@code f32.foo}). */
    private String typeIdent() {
        ws();
        int start = pos;
        if (!isIdentStart(charAt(0))) {
            throw error("expected type");
        }
        while (pos < src.length() && Character.isLetterOrDigit(src.charAt(pos))) {
            pos++;
        }
        return src.substring(start, pos);
    }

    static ElementType scalarType(String id) {
        FloatFormat ff = FloatFormat.byName(id);
        if (ff != null) {
            return new FloatType(ff);
        }
        var m = INT_TYPE.matcher(id);
        if (m.matches()) {
            int bits = Integer.parseInt(m.group(2));
            IntegerType.Signedness s =
                    switch (m.group(1)) {
                        case "si" -> IntegerType.Signedness.SIGNED;
                        case "ui" -> IntegerType.Signedness.UNSIGNED;
                        default -> IntegerType.Signedness.SIGNLESS;
                    };
            return new IntegerType(bits, s);
        }
        return null;
    }

    ElementType elementType() {
        int start = pos;
        Type t = type();
        if (t instanceof ElementType et) {
            return et;
        }
        throw errorAt(start, "expected an element type, got " + t);
    }

    private TensorType tensorType() {
        expect("<");
        List<Long> dims = new ArrayList<>();
        while (true) {
            ws();
            char c = charAt(0);
            if (c == '?') {
                pos++;
                dims.add(TensorType.DYNAMIC);
            } else if (Character.isDigit(c)) {
                int start = pos;
                while (Character.isDigit(charAt(0))) {
                    pos++;
                }
                dims.add(Long.parseLong(src.substring(start, pos)));
            } else {
                break;
            }
            if (charAt(0) != 'x') {
                throw error("expected 'x' after dimension");
            }
            pos++;
        }
        ElementType et = elementType();
        Attribute encoding = null;
        if (tryConsume(",")) {
            encoding = canonicalEncoding(attribute());
        }
        expect(">");
        return new TensorType(dims, et, encoding);
    }

    /**
     * {@code #stablehlo.type_extensions<bounds = [3, ?]>} is the long spelling of {@code
     * #stablehlo.bounds<3, ?>}; both denote the same tensor encoding.
     */
    private static Attribute canonicalEncoding(Attribute encoding) {
        if (encoding instanceof DialectAttr d
                && d.dialect().equals("stablehlo")
                && d.mnemonic().equals("type_extensions")
                && !d.hasParams()) {
            var m =
                    java.util.regex.Pattern.compile("bounds\\s*=\\s*\\[(.*)\\]")
                            .matcher(d.rawBody());
            if (m.matches()) {
                return DialectAttr.ofRaw(
                        "stablehlo",
                        "bounds",
                        m.group(1).trim().replaceAll("\\s*,\\s*", ", ").replaceAll("-\\d+", "?"));
            }
        }
        if (encoding instanceof DialectAttr d
                && d.dialect().equals("stablehlo")
                && d.mnemonic().equals("type_extensions")
                && d.hasParams()
                && d.params().size() == 1
                && d.param("bounds").isPresent()) {
            // all-integer bounds parse as an array parameter; -1 is the old spelling of '?'
            long[] bounds = com.qxotic.jota.stablehlo.op.OpSupport.toLongs(d.param("bounds").get());
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < bounds.length; i++) {
                sb.append(i > 0 ? ", " : "").append(bounds[i] < 0 ? "?" : Long.toString(bounds[i]));
            }
            return DialectAttr.ofRaw("stablehlo", "bounds", sb.toString());
        }
        if (encoding instanceof DialectAttr d
                && d.dialect().equals("stablehlo")
                && d.mnemonic().equals("bounds")
                && !d.hasParams()) {
            return DialectAttr.ofRaw(
                    "stablehlo", "bounds", d.rawBody().trim().replaceAll("\\s*,\\s*", ", "));
        }
        return encoding;
    }

    private FunctionType functionType() {
        expect("(");
        List<Type> inputs = new ArrayList<>();
        if (!tryConsume(")")) {
            do {
                inputs.add(type());
            } while (tryConsume(","));
            expect(")");
        }
        expect("->");
        return new FunctionType(inputs, resultTypes());
    }

    /** Results after {@code ->}: {@code (A, B)}, {@code ()} or a single non-parenthesized type. */
    List<Type> resultTypes() {
        ws();
        List<Type> results = new ArrayList<>();
        if (charAt(0) == '(') {
            int save = pos;
            pos++;
            if (tryConsume(")")) {
                if (peek("->")) { // a single function-typed result: () -> ...
                    pos = save;
                    results.add(type());
                }
                return results;
            }
            do {
                results.add(type());
            } while (tryConsume(","));
            expect(")");
            if (peek("->")) {
                pos = save;
                results.clear();
                results.add(type());
            }
            return results;
        }
        results.add(type());
        return results;
    }

    private Type dialectType() {
        int start = pos;
        expect("!");
        String name = suffixId();
        switch (name) {
            case "stablehlo.token" -> {
                return TokenType.INSTANCE;
            }
            case "stablehlo.future" -> {
                expect("<");
                List<Type> types = new ArrayList<>();
                do {
                    types.add(type());
                } while (tryConsume(","));
                expect(">");
                return new FutureType(types);
            }
            case "quant.uniform" -> {
                return quantizedType();
            }
            default -> {
                Type alias = typeAliases.get(name);
                if (alias != null) {
                    return alias;
                }
                if (charAt(0) == '<') {
                    balanced('<', '>');
                }
                return new OpaqueType(src.substring(start, pos));
            }
        }
    }

    private QuantizedType quantizedType() {
        expect("<");
        ElementType storage = elementType();
        if (!(storage instanceof IntegerType storageType)) {
            throw error("quantized storage type must be an integer type");
        }
        Long min = null;
        Long max = null;
        if (tryConsume("<")) {
            min = integer();
            expect(":");
            max = integer();
            expect(">");
        }
        expect(":");
        ElementType expressed = elementType();
        if (!(expressed instanceof FloatType expressedType)) {
            throw error("quantized expressed type must be a float type");
        }
        Integer dim = null;
        if (tryConsume(":")) {
            dim = (int) integer();
        }
        expect(",");
        List<Double> scales = new ArrayList<>();
        List<Long> zeroPoints = new ArrayList<>();
        boolean braces = tryConsume("{");
        do {
            scales.add(Double.parseDouble(numberToken()));
            zeroPoints.add(tryConsume(":") ? integer() : 0L);
        } while (braces && tryConsume(","));
        if (braces) {
            expect("}");
        }
        expect(">");
        return new QuantizedType(storageType, min, max, expressedType, dim, scales, zeroPoints);
    }

    // ================================================================== attributes

    /** Attribute dictionary {@code {a = 1, b, "c" = ...}}. */
    List<NamedAttribute> attrDict() {
        expect("{");
        List<NamedAttribute> entries = new ArrayList<>();
        if (tryConsume("}")) {
            return entries;
        }
        do {
            ws();
            String key = charAt(0) == '"' ? string() : ident();
            if (tryConsume("=")) {
                entries.add(new NamedAttribute(key, attribute()));
            } else {
                entries.add(new NamedAttribute(key, UnitAttr.INSTANCE));
            }
        } while (tryConsume(","));
        expect("}");
        return entries;
    }

    Attribute attribute() {
        ws();
        char c = charAt(0);
        if (c == '"') {
            return new StringAttr(string());
        }
        if (c == '[') {
            pos++;
            List<Attribute> elements = new ArrayList<>();
            if (!tryConsume("]")) {
                do {
                    elements.add(attribute());
                } while (tryConsume(","));
                expect("]");
            }
            return new ArrayAttr(elements);
        }
        if (c == '{') {
            return new DictionaryAttr(attrDict());
        }
        if (c == '@') {
            List<String> path = new ArrayList<>();
            path.add(symbolName());
            while (peek("::")) {
                pos += 2;
                path.add(symbolName());
            }
            return new SymbolRefAttr(path);
        }
        if (c == '#') {
            return hashAttribute();
        }
        if (peekNumber()) {
            return numberAttribute();
        }
        if (peekKeyword("true")) {
            pos += 4;
            return BoolAttr.TRUE;
        }
        if (peekKeyword("false")) {
            pos += 5;
            return BoolAttr.FALSE;
        }
        if (peekKeyword("unit")) {
            pos += 4;
            return UnitAttr.INSTANCE;
        }
        if (peekKeyword("dense")) {
            pos += 5;
            return denseElements();
        }
        if (peekKeyword("array")) {
            pos += 5;
            return denseArray();
        }
        if (peekType()) {
            return new TypeAttr(type());
        }
        if (peekIdent()) {
            int start = pos;
            String id = ident();
            if (charAt(0) == '<') { // dense_resource<...>, affine_map<...>, sparse<...>, ...
                balanced('<', '>');
                if (peek(":") && !peek("::")) {
                    pos++;
                    type();
                }
                return new OpaqueAttr(src.substring(start, pos));
            }
            return new OpaqueAttr(id);
        }
        throw error("expected attribute");
    }

    private Attribute numberAttribute() {
        int start = pos;
        String tok = numberToken();
        Type t = null;
        if (peek(":") && !peek("::")) {
            int save = pos;
            pos++;
            if (peekType()) {
                t = type();
            } else {
                pos = save;
            }
        }
        if (t instanceof FloatType ft) {
            return new FloatAttr(floatBits(tok, ft.format(), start), ft);
        }
        if (isFloatToken(tok)) {
            if (t != null) {
                throw errorAt(start, "float literal with non-float type " + t);
            }
            return new FloatAttr(FloatFormat.F64.encode(Double.parseDouble(tok)), FloatType.F64);
        }
        Type it = t == null ? IntegerType.I64 : t;
        long v = integerBits(parseInteger(tok), it, start);
        if (it instanceof IntegerType itype && itype.isBoolean()) {
            return new IntegerAttr(v, itype);
        }
        return new IntegerAttr(v, it);
    }

    private Attribute hashAttribute() {
        int start = pos;
        expect("#");
        String name = suffixId();
        int dot = name.indexOf('.');
        if (charAt(0) != '<') {
            Attribute alias = attributeAliases.get(name);
            if (alias != null) {
                return alias;
            }
            throw errorAt(start, "unknown attribute alias #" + name);
        }
        if (dot < 0) {
            // #dialect<kind VALUE> enum (or other dialect-specific body kept verbatim)
            String body = balanced('<', '>');
            String inner = body.substring(1, body.length() - 1).trim();
            String[] parts = inner.split("\\s+");
            if (parts.length == 2 && parts[0].matches("\\w+") && parts[1].matches("\\w+")) {
                return new EnumAttr(name, parts[0], parts[1]);
            }
            return new OpaqueAttr(src.substring(start, pos));
        }
        return dialectAttrBody(name.substring(0, dot), name.substring(dot + 1));
    }

    /**
     * Parses the {@code <...>} body of {@code #dialect.mnemonic<...>} at the cursor: a {@code key =
     * value} list becomes parameters, anything else is kept verbatim with whitespace runs collapsed
     * (so that formatting does not matter).
     */
    /**
     * Canonical parameters, so that every spelling MLIR accepts denotes the same attribute.
     *
     * <ul>
     *   <li>Parameters the StableHLO printer omits when they hold their default are made explicit:
     *       {@code #stablehlo.result_accuracy<mode = ...>} has zero {@code atol}, {@code rtol} and
     *       {@code ulps}.
     *   <li>Empty dimension lists of {@code #stablehlo.dot / gather / scatter}, which MLIR omits,
     *       are dropped.
     *   <li>Struct parameters MLIR prints with their prefix elided are spelled in full: the {@code
     *       axes} of {@code #stablehlo.mesh<axes = [<name = "x", size = 2>]>} are {@code
     *       #stablehlo.mesh_axis<...>}.
     * </ul>
     */
    private static List<NamedAttribute> withDefaults(
            String dialect, String mnemonic, List<NamedAttribute> params) {
        if (dialect.equals("stablehlo")
                && java.util.Set.of("dot", "gather", "scatter").contains(mnemonic)) {
            // MLIR omits empty dimension lists: an explicit [] is the same attribute
            List<NamedAttribute> out = new ArrayList<>();
            for (NamedAttribute p : params) {
                if (!(p.value() instanceof DenseArrayAttr a && a.size() == 0)) {
                    out.add(p);
                }
            }
            return out;
        }
        if (dialect.equals("stablehlo") && mnemonic.equals("mesh")) {
            List<NamedAttribute> out = new ArrayList<>();
            for (NamedAttribute p : params) {
                if (p.name().equals("axes") && p.value() instanceof ArrayAttr axes) {
                    List<Attribute> full = new ArrayList<>();
                    for (Attribute a : axes.elements()) {
                        full.add(
                                a instanceof DictionaryAttr d
                                        ? DialectAttr.ofParams(
                                                "stablehlo", "mesh_axis", d.entries())
                                        : a);
                    }
                    p = new NamedAttribute("axes", new ArrayAttr(full));
                }
                out.add(p);
            }
            return out;
        }
        if (!dialect.equals("stablehlo") || !mnemonic.equals("result_accuracy")) {
            return params;
        }
        List<NamedAttribute> out = new ArrayList<>(params);
        java.util.Set<String> present = new java.util.HashSet<>();
        params.forEach(p -> present.add(p.name()));
        if (!present.contains("atol")) {
            out.add(new NamedAttribute("atol", FloatAttr.of(0.0, FloatType.F64)));
        }
        if (!present.contains("rtol")) {
            out.add(new NamedAttribute("rtol", FloatAttr.of(0.0, FloatType.F64)));
        }
        if (!present.contains("ulps")) {
            out.add(new NamedAttribute("ulps", IntegerAttr.i64(0)));
        }
        return out;
    }

    DialectAttr dialectAttrBody(String dialect, String mnemonic) {
        ws();
        int bodyStart = pos;
        String body = balanced('<', '>');
        List<NamedAttribute> params = tryParseParams(bodyStart + 1, pos - 1);
        if (params != null) {
            return DialectAttr.ofParams(dialect, mnemonic, withDefaults(dialect, mnemonic, params));
        }
        String inner = body.substring(1, body.length() - 1).trim().replaceAll("\\s+", " ");
        DialectAttr raw = DialectAttr.ofRaw(dialect, mnemonic, inner);
        if (dialect.equals("stablehlo") && mnemonic.equals("conv")) {
            // both spellings (compact layout, raw fields) denote the same dimension numbers:
            // keep the one MLIR prints
            try {
                return ConvDimensionNumbers.from(raw).toAttribute();
            } catch (IllegalArgumentException e) {
                return raw; // malformed: left to the verifier
            }
        }
        return raw;
    }

    /**
     * Parses {@code key = value, ...} between [from, to); returns null when the body is not a
     * parameter list.
     */
    private List<NamedAttribute> tryParseParams(int from, int to) {
        int save = pos;
        pos = from;
        try {
            List<NamedAttribute> params = new ArrayList<>();
            ws();
            if (pos >= to) {
                return params; // empty body: no parameters
            }
            do {
                ws();
                if (pos >= to) {
                    break; // trailing comma
                }
                String key = ident();
                expect("=");
                params.add(new NamedAttribute(key, paramValue()));
                ws();
            } while (pos < to && tryConsume(","));
            ws();
            return pos == to ? params : null;
        } catch (ParseException e) {
            return null;
        } finally {
            pos = save;
        }
    }

    /** Value of a dialect attribute parameter: integers/lists print without types. */
    private Attribute paramValue() {
        ws();
        char c = charAt(0);
        if (c == '<') { // anonymous nested struct: <key = value, ...>
            pos++;
            List<NamedAttribute> entries = new ArrayList<>();
            if (!tryConsume(">")) {
                do {
                    String key = ident();
                    expect("=");
                    entries.add(new NamedAttribute(key, paramValue()));
                } while (tryConsume(","));
                expect(">");
            }
            return new DictionaryAttr(entries);
        }
        if (c == '(') { // sub-axis info of an axis reference: (pre_size)size
            pos++;
            long preSize = integer();
            expect(")");
            long size = integer();
            return DialectAttr.ofParams(
                    "stablehlo",
                    "sub_axis_info",
                    List.of(
                            new NamedAttribute("pre_size", IntegerAttr.i64(preSize)),
                            new NamedAttribute("size", IntegerAttr.i64(size))));
        }
        if (c == '[') {
            int save = pos;
            pos++;
            List<Long> ints = new ArrayList<>();
            boolean allInts = true;
            if (!tryConsume("]")) {
                do {
                    if (peekNumber()) {
                        String tok = numberToken();
                        if (isFloatToken(tok)) {
                            allInts = false;
                            break;
                        }
                        ints.add(parseInteger(tok).longValue());
                    } else {
                        allInts = false;
                        break;
                    }
                } while (tryConsume(","));
                if (allInts) {
                    expect("]");
                }
            }
            if (allInts) {
                return DenseArrayAttr.i64(ints.stream().mapToLong(Long::longValue).toArray());
            }
            pos = save + 1;
            List<Attribute> elements = new ArrayList<>();
            do {
                elements.add(paramValue());
            } while (tryConsume(","));
            expect("]");
            return new ArrayAttr(elements);
        }
        if (peekNumber()) {
            String tok = numberToken();
            if (isFloatToken(tok)) {
                return FloatAttr.of(Double.parseDouble(tok), FloatType.F64);
            }
            return IntegerAttr.i64(parseInteger(tok).longValue());
        }
        if (peekKeyword("true")) {
            pos += 4;
            return BoolAttr.TRUE;
        }
        if (peekKeyword("false")) {
            pos += 5;
            return BoolAttr.FALSE;
        }
        if (peekType()) {
            return new TypeAttr(type());
        }
        if (peekIdent()) {
            String id = ident();
            if (charAt(0) == '<') {
                return new OpaqueAttr(id + balanced('<', '>'));
            }
            return new OpaqueAttr(id);
        }
        return attribute();
    }

    private DenseArrayAttr denseArray() {
        expect("<");
        Type et = type();
        List<Long> values = new ArrayList<>();
        if (tryConsume(":")) {
            do {
                int at = pos;
                ws();
                if (tryKeyword("true")) {
                    values.add(1L);
                } else if (tryKeyword("false")) {
                    values.add(0L);
                } else if (et instanceof FloatType ft) {
                    values.add(floatBits(numberToken(), ft.format(), at));
                } else {
                    values.add(integerBits(parseInteger(numberToken()), et, at));
                }
            } while (tryConsume(","));
        }
        expect(">");
        return new DenseArrayAttr(et, values.stream().mapToLong(Long::longValue).toArray());
    }

    // ================================================================== dense literals

    /** One element literal: a raw token, or a complex pair. */
    private record Elem(String token, String imag, int at) {}

    private DenseElementsAttr denseElements() {
        int start = pos;
        expect("<");
        List<Elem> elems = new ArrayList<>();
        byte[] blob = null;
        boolean scalarLiteral = false;
        ws();
        if (charAt(0) == '"') {
            String hex = string();
            if (!hex.startsWith("0x") && !hex.startsWith("0X")) {
                throw errorAt(start, "expected hexadecimal dense blob");
            }
            blob = hexBytes(hex.substring(2), start);
        } else if (charAt(0) == '[') {
            literalList(elems);
        } else if (charAt(0) != '>') {
            elems.add(elementLiteral());
            scalarLiteral = true;
        }
        expect(">");
        expect(":");
        int typeAt = pos;
        Type t = type();
        if (!(t instanceof TensorType tt)) {
            throw errorAt(typeAt, "dense elements need a tensor type, got " + t);
        }
        int elemBytes = DenseElementsAttr.storageBytes(tt);
        if (blob != null) {
            return fromBlob(tt, blob, elemBytes, typeAt);
        }
        long n = tt.numElements();
        boolean splat = scalarLiteral || (elems.size() == 1 && n != 1);
        if (!splat && elems.size() != n) {
            throw errorAt(start, "dense literal has " + elems.size() + " elements, type " + tt);
        }
        int count = splat ? 1 : elems.size();
        byte[] data = new byte[count * elemBytes];
        for (int i = 0; i < count; i++) {
            encodeElement(elems.get(i), tt.elementType(), data, i * elemBytes);
        }
        return new DenseElementsAttr(tt, data, splat);
    }

    private void literalList(List<Elem> out) {
        expect("[");
        if (tryConsume("]")) {
            return;
        }
        do {
            ws();
            if (charAt(0) == '[') {
                literalList(out);
            } else {
                out.add(elementLiteral());
            }
        } while (tryConsume(","));
        expect("]");
    }

    private Elem elementLiteral() {
        ws();
        int at = pos;
        if (charAt(0) == '(') {
            pos++;
            String re = scalarToken();
            expect(",");
            String im = scalarToken();
            expect(")");
            return new Elem(re, im, at);
        }
        return new Elem(scalarToken(), null, at);
    }

    private String scalarToken() {
        ws();
        if (peekKeyword("true")) {
            pos += 4;
            return "true";
        }
        if (peekKeyword("false")) {
            pos += 5;
            return "false";
        }
        for (String special : new String[] {"inf", "-inf", "+inf", "nan", "-nan"}) {
            if (peekKeyword(special)) {
                pos += special.length();
                return special;
            }
        }
        return numberToken();
    }

    private void encodeElement(Elem e, ElementType et, byte[] data, int offset) {
        switch (et) {
            case ComplexType ct -> {
                if (e.imag == null) {
                    throw errorAt(e.at, "expected complex literal (re, im)");
                }
                int half = ct.elementType().storageBytes();
                FloatFormat f = ct.elementType().format();
                writeLE(data, offset, half, floatBits(e.token, f, e.at));
                writeLE(data, offset + half, half, floatBits(e.imag, f, e.at));
            }
            case FloatType ft ->
                    writeLE(data, offset, ft.storageBytes(), floatBits(e.token, ft.format(), e.at));
            case IntegerType it -> {
                long bits;
                if (e.token.equals("true") || e.token.equals("false")) {
                    if (!it.isBoolean()) {
                        throw errorAt(e.at, "boolean literal for " + it);
                    }
                    bits = e.token.equals("true") ? 1 : 0;
                } else {
                    bits = integerBits(parseInteger(e.token), it, e.at);
                }
                writeLE(data, offset, it.storageBytes(), bits);
            }
            case IndexType it ->
                    writeLE(data, offset, 8, integerBits(parseInteger(e.token), it, e.at));
            case QuantizedType qt -> {
                long bits = integerBits(parseInteger(e.token), qt.storageType(), e.at);
                writeLE(data, offset, qt.storageBytes(), bits);
            }
        }
    }

    private DenseElementsAttr fromBlob(TensorType tt, byte[] blob, int elemBytes, int at) {
        long n = tt.numElements();
        if (blob.length == elemBytes && n != 1) {
            return new DenseElementsAttr(tt, blob, true);
        }
        if (blob.length == n * elemBytes) {
            return new DenseElementsAttr(tt, blob, false);
        }
        if (tt.elementType() instanceof IntegerType it
                && it.isBoolean()
                && blob.length == (n + 7) / 8) { // bit-packed i1
            byte[] bytes = new byte[Math.toIntExact(n)];
            for (int i = 0; i < n; i++) {
                bytes[i] = (byte) ((blob[i / 8] >> (i % 8)) & 1);
            }
            return new DenseElementsAttr(tt, bytes, false);
        }
        throw errorAt(at, "hex blob of " + blob.length + " bytes does not fit " + tt);
    }

    private byte[] hexBytes(String hex, int at) {
        if (hex.length() % 2 != 0) {
            throw errorAt(at, "odd number of hex digits");
        }
        byte[] out = new byte[hex.length() / 2];
        for (int i = 0; i < out.length; i++) {
            int hi = Character.digit(hex.charAt(2 * i), 16);
            int lo = Character.digit(hex.charAt(2 * i + 1), 16);
            if (hi < 0 || lo < 0) {
                throw errorAt(at, "bad hex digit");
            }
            out[i] = (byte) (hi * 16 + lo);
        }
        return out;
    }

    static void writeLE(byte[] buf, int off, int n, long v) {
        for (int i = 0; i < n; i++) {
            buf[off + i] = (byte) (v >>> (8 * i));
        }
    }

    /**
     * Bits of a float literal: hexadecimal is the raw pattern, decimal is rounded to the format.
     */
    long floatBits(String tok, FloatFormat format, int at) {
        String t = tok;
        switch (t) {
            case "inf", "+inf" -> {
                return format.encode(Double.POSITIVE_INFINITY);
            }
            case "-inf" -> {
                return format.encode(Double.NEGATIVE_INFINITY);
            }
            case "nan", "-nan" -> {
                return format.nanBits();
            }
            default -> {}
        }
        if (t.startsWith("0x") || t.startsWith("0X")) {
            BigInteger v = new BigInteger(t.substring(2), 16);
            if (v.bitLength() > format.bitWidth()) {
                throw errorAt(at, "hex literal too wide for " + format);
            }
            return v.longValue();
        }
        try {
            if (format == FloatFormat.F32) {
                return Float.floatToRawIntBits(Float.parseFloat(t)) & 0xFFFFFFFFL;
            }
            return format.encode(Double.parseDouble(t));
        } catch (NumberFormatException e) {
            throw errorAt(at, "bad float literal '" + tok + "'");
        }
    }

    /** Two's complement bits of an integer literal, checking the range of {@code type}. */
    long integerBits(BigInteger v, Type type, int at) {
        int bits =
                switch (type) {
                    case IntegerType it -> it.bitWidth();
                    case IndexType ignored -> 64;
                    default -> throw errorAt(at, "integer literal for non-integer type " + type);
                };
        BigInteger min = BigInteger.ONE.shiftLeft(bits - 1).negate();
        BigInteger max = BigInteger.ONE.shiftLeft(bits).subtract(BigInteger.ONE);
        if (bits == 1) {
            min = BigInteger.valueOf(-1);
        }
        if (v.compareTo(min) < 0 || v.compareTo(max) > 0) {
            throw errorAt(at, "integer literal " + v + " out of range for " + type);
        }
        long raw = v.longValue();
        return bits == 64 ? raw : raw & ((1L << bits) - 1);
    }
}
