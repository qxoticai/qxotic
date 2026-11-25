package com.qxotic.jota.stablehlo.vhlo;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.attr.UnitAttr;
import com.qxotic.jota.stablehlo.ir.Block;
import com.qxotic.jota.stablehlo.ir.Operation;
import com.qxotic.jota.stablehlo.ir.Program;
import com.qxotic.jota.stablehlo.ir.Region;
import com.qxotic.jota.stablehlo.ir.Value;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Legalizes a VHLO program (as read from a portable artifact) to StableHLO, following {@code
 * stablehlo/transforms/VhloLegalizeToStablehlo.cpp}.
 *
 * <p>Every VHLO version of an op ({@code gather_v1}, {@code gather_v2}...) maps to the current
 * StableHLO op: attributes added by later versions are simply absent in older ones and take their
 * default, which is what StableHLO's upgrade patterns ({@code VhloToVersionPatterns.td}) produce.
 * Then, per op: attributes equal to their StableHLO default are dropped ({@code removeDefaults}),
 * flattened dimension numbers are imploded into {@code #stablehlo.gather<...>}, {@code
 * #stablehlo.conv<...>}... ({@code implodeSpecial}), and integer tensors become {@code array<i64>}
 * where StableHLO declares a dense array ({@code convertSpecial}).
 *
 * <p>VHLO attributes and types were already decoded into their StableHLO spelling by the bytecode
 * reader.
 */
public final class VhloToStableHlo {

    private static final Pattern VERSIONED = Pattern.compile("vhlo\\.(.+)_v(\\d+)");

    private static final Set<String> COLLECTIVES =
            Set.of(
                    "all_gather",
                    "all_reduce",
                    "all_to_all",
                    "collective_broadcast",
                    "collective_permute",
                    "collective_reduce",
                    "reduce_scatter");

    private static final Set<String> RESULT_ACCURACY_OPS =
            Set.of(
                    "cbrt",
                    "cosine",
                    "exponential",
                    "exponential_minus_one",
                    "exp2",
                    "log",
                    "log_plus_one",
                    "log2",
                    "logistic",
                    "rsqrt",
                    "sine",
                    "sqrt",
                    "tan",
                    "tanh");

    /** Attributes StableHLO declares as {@code array<i64>} although VHLO stores a tensor. */
    private static final Map<String, Set<String>> I64_ARRAYS =
            Map.ofEntries(
                    Map.entry("fft", Set.of("fft_length")),
                    Map.entry("broadcast", Set.of("broadcast_sizes")),
                    Map.entry("dynamic_slice", Set.of("slice_sizes")),
                    Map.entry("gather", Set.of("slice_sizes")),
                    Map.entry("map", Set.of("dimensions")),
                    Map.entry("reduce", Set.of("dimensions")),
                    Map.entry("reverse", Set.of("dimensions")),
                    Map.entry("transpose", Set.of("permutation")),
                    Map.entry(
                            "pad",
                            Set.of("edge_padding_low", "edge_padding_high", "interior_padding")),
                    Map.entry("slice", Set.of("start_indices", "limit_indices", "strides")),
                    Map.entry("broadcast_in_dim", Set.of("broadcast_dimensions")),
                    Map.entry(
                            "dynamic_broadcast_in_dim",
                            Set.of(
                                    "broadcast_dimensions",
                                    "known_expanding_dimensions",
                                    "known_nonexpanding_dimensions")),
                    Map.entry("select_and_scatter", Set.of("window_dimensions", "window_strides")),
                    Map.entry(
                            "reduce_window",
                            Set.of(
                                    "window_dimensions",
                                    "window_strides",
                                    "base_dilations",
                                    "window_dilations")),
                    Map.entry(
                            "convolution",
                            Set.of("window_strides", "lhs_dilation", "rhs_dilation")),
                    Map.entry(
                            "dynamic_conv",
                            Set.of("window_strides", "lhs_dilation", "rhs_dilation")));

    private static final List<String> CUSTOM_CALL_API_VERSIONS =
            List.of(
                    "API_VERSION_UNSPECIFIED",
                    "API_VERSION_ORIGINAL",
                    "API_VERSION_STATUS_RETURNING",
                    "API_VERSION_STATUS_RETURNING_UNIFIED",
                    "API_VERSION_TYPED_FFI");

    private final Map<Value, Value> values = new IdentityHashMap<>();

    private VhloToStableHlo() {}

    /** Converts a VHLO program to StableHLO. */
    public static Program convert(Program vhlo) {
        return new Program(new VhloToStableHlo().convert(vhlo.operation(), false));
    }

    private Operation convert(Operation op, boolean parentIsFunc) {
        Matcher m = VERSIONED.matcher(op.name());
        String base = m.matches() ? m.group(1) : null;
        String name;
        List<NamedAttribute> inherent;
        List<NamedAttribute> discardable;
        if (base == null) {
            if (op.name().startsWith("vhlo.")) {
                throw new VhloException("unknown VHLO operation " + op.name());
            }
            name = op.name();
            inherent = op.properties();
            discardable = op.attributes();
        } else {
            VhloOpSchemas.Schema schema = VhloOpSchemas.get(op.mnemonic());
            if (schema == null) {
                throw new VhloException("unknown VHLO operation " + op.name());
            }
            name =
                    switch (base) {
                        case "func" -> "func.func";
                        case "call" -> "func.call";
                        case "return" -> parentIsFunc ? "func.return" : "stablehlo.return";
                        default -> "stablehlo." + base;
                    };
            Set<String> inherentNames = Set.copyOf(schema.attributes());
            inherent = new ArrayList<>();
            discardable = new ArrayList<>();
            for (NamedAttribute a : op.allAttributes()) {
                (inherentNames.contains(a.name()) ? inherent : discardable).add(a);
            }
            inherent = legalizeAttributes(base, inherent);
        }
        List<Value> operands = new ArrayList<>(op.numOperands());
        for (Value v : op.operands()) {
            Value mapped = values.get(v);
            if (mapped == null) {
                throw new VhloException("operand defined after its use in " + op.name());
            }
            operands.add(mapped);
        }
        List<Region> regions = new ArrayList<>();
        for (int i = 0; i < op.regions().size(); i++) {
            regions.add(new Region());
        }
        Operation out =
                Operation.create(name, operands, op.resultTypes(), inherent, discardable, regions);
        for (int i = 0; i < op.numResults(); i++) {
            values.put(op.result(i), out.result(i));
        }
        boolean isFunc = name.equals("func.func");
        for (int i = 0; i < op.regions().size(); i++) {
            for (Block b : op.region(i).blocks()) {
                Block nb = new Block();
                for (Value arg : b.arguments()) {
                    values.put(arg, nb.addArgument(arg.type(), null));
                }
                for (Operation inner : b.operations()) {
                    nb.addOperation(convert(inner, isFunc));
                }
                regions.get(i).addBlock(nb);
            }
        }
        return out;
    }

    // ================================================================== attributes

    private static List<NamedAttribute> legalizeAttributes(String op, List<NamedAttribute> in) {
        Map<String, Attribute> a = new LinkedHashMap<>();
        for (NamedAttribute n : in) {
            a.put(n.name(), n.value());
        }
        removeDefaults(op, a);
        List<NamedAttribute> out = new ArrayList<>();
        implode(op, a, out);
        Set<String> i64Arrays = I64_ARRAYS.getOrDefault(op, Set.of());
        for (Map.Entry<String, Attribute> e : a.entrySet()) {
            String n = e.getKey();
            Attribute v = e.getValue();
            if (COLLECTIVES.contains(op) && n.equals("channel_id")) {
                out.add(new NamedAttribute("channel_handle", channelHandle(toLong(v), 0)));
            } else if (COLLECTIVES.contains(op)
                    && (n.equals("use_global_device_ids") || n.equals("has_dynamic_root"))) {
                if (isBool(v, true)) {
                    out.add(new NamedAttribute(n, UnitAttr.INSTANCE));
                }
            } else if (op.equals("custom_call") && n.equals("called_computations")) {
                List<Attribute> symbols = new ArrayList<>();
                for (Attribute s : ((ArrayAttr) v).elements()) {
                    symbols.add(SymbolRefAttr.of(((StringAttr) s).value()));
                }
                out.add(new NamedAttribute(n, new ArrayAttr(symbols)));
            } else if (op.equals("custom_call") && n.equals("api_version")) {
                out.add(
                        new NamedAttribute(
                                n,
                                IntegerAttr.i32(
                                        CUSTOM_CALL_API_VERSIONS.indexOf(((EnumAttr) v).value()))));
            } else if ((op.equals("composite") && n.equals("decomposition"))
                    || (op.equals("call") && n.equals("callee"))) {
                out.add(new NamedAttribute(n, SymbolRefAttr.of(((StringAttr) v).value())));
            } else if (i64Arrays.contains(n)) {
                out.add(new NamedAttribute(n, DenseArrayAttr.i64(longs(v))));
            } else if ((op.equals("convolution") || op.equals("dynamic_conv"))
                    && n.equals("window_reversal")) {
                long[] bits = longs(v);
                boolean[] b = new boolean[bits.length];
                for (int i = 0; i < b.length; i++) {
                    b[i] = bits[i] != 0;
                }
                out.add(new NamedAttribute(n, DenseArrayAttr.i1(b)));
            } else {
                out.add(new NamedAttribute(n, v));
            }
        }
        return out;
    }

    /** Port of {@code removeDefaults}: drop attributes equal to their StableHLO default. */
    private static void removeDefaults(String op, Map<String, Attribute> a) {
        switch (op) {
            case "func" -> {
                removeIf(a, "sym_visibility", v -> isString(v, ""));
                removeIf(a, "arg_attrs", VhloToStableHlo::isEmptyArray);
                removeIf(a, "res_attrs", VhloToStableHlo::isEmptyArray);
            }
            case "cholesky" -> removeIf(a, "lower", v -> isBool(v, false));
            case "compare" -> removeIf(a, "compare_type", v -> isEnum(v, "NOTYPE"));
            case "composite" -> {
                removeIf(a, "version", v -> isInteger(v, 0));
                removeIf(a, "composite_attributes", VhloToStableHlo::isEmptyDictionary);
            }
            case "convolution", "dynamic_conv" -> {
                removeIf(a, "window_strides", v -> isSplat(v, 1));
                if (op.equals("convolution")) {
                    removeIf(a, "padding", v -> isSplat(v, 0));
                } else {
                    a.remove("padding"); // dynamic_conv_v1 only: the operand carries the padding
                }
                removeIf(a, "lhs_dilation", v -> isSplat(v, 1));
                removeIf(a, "rhs_dilation", v -> isSplat(v, 1));
                removeIf(a, "window_reversal", v -> isSplat(v, 0));
            }
            case "custom_call" -> {
                removeIf(a, "has_side_effect", v -> isBool(v, false));
                removeIf(a, "backend_config", v -> isString(v, "") || isEmptyDictionary(v));
                removeIf(a, "api_version", v -> isEnum(v, "API_VERSION_ORIGINAL"));
                removeIf(a, "called_computations", VhloToStableHlo::isEmptyArray);
                if (isEmptyArray(a.get("operand_layouts"))
                        && isEmptyArray(a.get("result_layouts"))) {
                    a.remove("operand_layouts");
                    a.remove("result_layouts");
                }
                removeIf(a, "output_operand_aliases", VhloToStableHlo::isEmptyArray);
                removeIf(a, "result_tilings", VhloToStableHlo::isEmptyArray);
            }
            case "dynamic_broadcast_in_dim" -> {
                removeIf(a, "known_expanding_dimensions", VhloToStableHlo::isEmptyTensor);
                removeIf(a, "known_nonexpanding_dimensions", VhloToStableHlo::isEmptyTensor);
            }
            case "infeed" -> {
                removeIf(a, "infeed_config", v -> isString(v, ""));
                removeIf(a, "layout", VhloToStableHlo::isEmptyArray);
            }
            case "outfeed" -> removeIf(a, "outfeed_config", v -> isString(v, ""));
            case "send", "recv" -> {
                removeIf(a, "is_host_transfer", v -> isBool(v, false));
                removeIf(a, "source_target_pairs", VhloToStableHlo::isEmptyTensor);
            }
            case "reduce_window", "select_and_scatter" -> {
                removeIf(a, "window_strides", v -> isSplat(v, 1));
                if (op.equals("reduce_window")) {
                    removeIf(a, "base_dilations", v -> isSplat(v, 1));
                    removeIf(a, "window_dilations", v -> isSplat(v, 1));
                }
                removeIf(a, "padding", v -> isSplat(v, 0));
            }
            case "scatter" -> removeIf(a, "unique_indices", v -> isBool(v, false));
            case "sort" -> {
                removeIf(a, "dimension", v -> isInteger(v, -1));
                removeIf(a, "is_stable", v -> isBool(v, false));
            }
            default -> {}
        }
        if (COLLECTIVES.contains(op)) {
            removeIf(a, "channel_id", v -> isInteger(v, 0));
            removeIf(a, "use_global_device_ids", v -> isBool(v, false));
            removeIf(a, "has_dynamic_root", v -> isBool(v, false));
        }
        if (Set.of("convolution", "dynamic_conv", "dot_general", "dot").contains(op)) {
            removeIf(a, "precision_config", v -> isSplatArray(v, "DEFAULT"));
        }
        if (Set.of("gather", "dynamic_gather", "scatter").contains(op)) {
            removeIf(a, "indices_are_sorted", v -> isBool(v, false));
        }
        if (RESULT_ACCURACY_OPS.contains(op)) {
            removeIf(a, "result_accuracy", VhloToStableHlo::isDefaultResultAccuracy);
        }
    }

    /** Port of {@code implodeSpecial}: flattened VHLO attributes back to StableHLO structs. */
    private static void implode(String op, Map<String, Attribute> a, List<NamedAttribute> out) {
        switch (op) {
            case "convolution", "dynamic_conv" ->
                    out.add(
                            new NamedAttribute(
                                    "dimension_numbers",
                                    new ConvDimensionNumbers(
                                                    toLong(take(a, "input_batch_dimension")),
                                                    toLong(take(a, "input_feature_dimension")),
                                                    longs(take(a, "input_spatial_dimensions")),
                                                    toLong(
                                                            take(
                                                                    a,
                                                                    "kernel_input_feature_dimension")),
                                                    toLong(
                                                            take(
                                                                    a,
                                                                    "kernel_output_feature_dimension")),
                                                    longs(take(a, "kernel_spatial_dimensions")),
                                                    toLong(take(a, "output_batch_dimension")),
                                                    toLong(take(a, "output_feature_dimension")),
                                                    longs(take(a, "output_spatial_dimensions")))
                                            .toAttribute()));
            case "dot_general" -> {
                out.add(
                        new NamedAttribute(
                                "dot_dimension_numbers",
                                new DotDimensionNumbers(
                                                longs(take(a, "lhs_batching_dimensions")),
                                                longs(take(a, "rhs_batching_dimensions")),
                                                longs(take(a, "lhs_contracting_dimensions")),
                                                longs(take(a, "rhs_contracting_dimensions")))
                                        .toAttribute()));
                List<String> algorithmFields =
                        List.of(
                                "lhs_precision_type",
                                "rhs_precision_type",
                                "accumulation_type",
                                "lhs_component_count",
                                "rhs_component_count",
                                "num_primitive_operations",
                                "allow_imprecise_accumulation");
                List<NamedAttribute> params = new ArrayList<>();
                for (String f : algorithmFields) {
                    Attribute v = a.remove(f);
                    if (v != null && !isNone(v)) {
                        params.add(
                                new NamedAttribute(
                                        f,
                                        v instanceof IntegerAttr i
                                                ? IntegerAttr.i64(i.value())
                                                : v));
                    }
                }
                if (!params.isEmpty()) {
                    out.add(
                            new NamedAttribute(
                                    "algorithm",
                                    DialectAttr.ofParams("stablehlo", "dot_algorithm", params)));
                }
                Attribute blockScaling = a.remove("block_scaling_config");
                if (blockScaling != null && !isNone(blockScaling)) {
                    out.add(
                            new NamedAttribute(
                                    "block_scaling_config",
                                    perOperandConfig(
                                            "block_scaling_config",
                                            blockScaling,
                                            List.of("scale_idx"),
                                            List.of("zero_idx"))));
                }
                Attribute sparsity = a.remove("sparsity_config");
                if (sparsity != null && !isNone(sparsity)) {
                    out.add(
                            new NamedAttribute(
                                    "sparsity_config",
                                    perOperandConfig(
                                            "sparsity_config",
                                            sparsity,
                                            List.of(
                                                    "num_non_zero",
                                                    "block_size",
                                                    "dimension",
                                                    "stride",
                                                    "idx"),
                                            List.of())));
                }
            }
            case "gather", "dynamic_gather" ->
                    out.add(
                            new NamedAttribute(
                                    "dimension_numbers",
                                    new GatherDimensionNumbers(
                                                    longs(take(a, "offset_dims")),
                                                    longs(take(a, "collapsed_slice_dims")),
                                                    longsOrEmpty(a.remove("operand_batching_dims")),
                                                    longsOrEmpty(
                                                            a.remove(
                                                                    "start_indices_batching_dims")),
                                                    longs(take(a, "start_index_map")),
                                                    toLong(take(a, "index_vector_dim")))
                                            .toAttribute()));
            case "scatter" ->
                    out.add(
                            new NamedAttribute(
                                    "scatter_dimension_numbers",
                                    new ScatterDimensionNumbers(
                                                    longs(take(a, "update_window_dims")),
                                                    longs(take(a, "inserted_window_dims")),
                                                    longsOrEmpty(a.remove("input_batching_dims")),
                                                    longsOrEmpty(
                                                            a.remove(
                                                                    "scatter_indices_batching_dims")),
                                                    longs(take(a, "scatter_dims_to_operand_dims")),
                                                    toLong(take(a, "index_vector_dim")))
                                            .toAttribute()));
            case "send", "recv" ->
                    out.add(
                            new NamedAttribute(
                                    "channel_handle",
                                    channelHandle(
                                            toLong(take(a, "channel_id")),
                                            toLong(take(a, "channel_type")))));
            default -> {}
        }
    }

    /**
     * {@code {lhs = {...}, rhs = {...}}} (a VHLO dictionary) to {@code #stablehlo.<mnemonic><lhs =
     * <...>, rhs = <...>>}: integers become untyped scalars (the {@code required} ones default to
     * 0), integer lists become {@code [..]} and are omitted when empty, {@code optional} scalars
     * are omitted when absent.
     */
    private static DialectAttr perOperandConfig(
            String mnemonic, Attribute config, List<String> required, List<String> optional) {
        if (!(config instanceof DictionaryAttr dict)) {
            throw new VhloException(mnemonic + " is not a dictionary: " + config);
        }
        List<NamedAttribute> sides = new ArrayList<>();
        for (String side : List.of("lhs", "rhs")) {
            if (!(dict.get(side).orElse(null) instanceof DictionaryAttr s)) {
                continue;
            }
            List<NamedAttribute> fields = new ArrayList<>();
            for (NamedAttribute e : s.entries()) {
                Attribute v = e.value();
                if (v instanceof IntegerAttr i) {
                    fields.add(new NamedAttribute(e.name(), IntegerAttr.i64(i.value())));
                } else if (v instanceof ArrayAttr arr) {
                    if (!arr.elements().isEmpty()) {
                        long[] ints =
                                arr.elements().stream()
                                        .mapToLong(VhloToStableHlo::toLong)
                                        .toArray();
                        fields.add(new NamedAttribute(e.name(), DenseArrayAttr.i64(ints)));
                    }
                } else {
                    fields.add(e);
                }
            }
            for (String r : required) {
                if (s.get(r).isEmpty()) {
                    fields.add(new NamedAttribute(r, IntegerAttr.i64(0)));
                }
            }
            sides.add(new NamedAttribute(side, new DictionaryAttr(fields)));
        }
        return DialectAttr.ofParams("stablehlo", mnemonic, sides);
    }

    // ================================================================== helpers

    private static Attribute take(Map<String, Attribute> a, String name) {
        Attribute v = a.remove(name);
        if (v == null) {
            throw new VhloException("missing VHLO attribute '" + name + "'");
        }
        return v;
    }

    private static void removeIf(
            Map<String, Attribute> a, String name, java.util.function.Predicate<Attribute> p) {
        Attribute v = a.get(name);
        if (v != null && p.test(v)) {
            a.remove(name);
        }
    }

    private static DialectAttr channelHandle(long handle, long type) {
        return DialectAttr.ofParams(
                "stablehlo",
                "channel_handle",
                List.of(
                        new NamedAttribute("handle", IntegerAttr.i64(handle)),
                        new NamedAttribute("type", IntegerAttr.i64(type))));
    }

    private static long toLong(Attribute v) {
        if (v instanceof IntegerAttr i) {
            return i.value();
        }
        throw new VhloException("expected an integer attribute, got " + v);
    }

    private static long[] longs(Attribute v) {
        if (v instanceof DenseElementsAttr d) {
            long[] out = new long[Math.toIntExact(d.numElements())];
            for (int i = 0; i < out.length; i++) {
                out[i] = d.getLong(i);
            }
            return out;
        }
        if (v instanceof DenseArrayAttr d) {
            return d.values();
        }
        throw new VhloException("expected an integer tensor attribute, got " + v);
    }

    private static long[] longsOrEmpty(Attribute v) {
        return v == null ? new long[0] : longs(v);
    }

    private static boolean isNone(Attribute v) {
        return v instanceof TypeAttr t
                && t.type() instanceof OpaqueType o
                && o.text().equals("none");
    }

    private static boolean isBool(Attribute v, boolean value) {
        return v instanceof BoolAttr b && b.value() == value;
    }

    private static boolean isString(Attribute v, String value) {
        return v instanceof StringAttr s && s.value().equals(value);
    }

    private static boolean isInteger(Attribute v, long value) {
        return v instanceof IntegerAttr i && i.value() == value;
    }

    private static boolean isEnum(Attribute v, String value) {
        return v instanceof EnumAttr e && e.value().equals(value);
    }

    private static boolean isEmptyArray(Attribute v) {
        return v instanceof ArrayAttr a && a.elements().isEmpty();
    }

    private static boolean isEmptyDictionary(Attribute v) {
        return v instanceof DictionaryAttr d && d.isEmpty();
    }

    private static boolean isEmptyTensor(Attribute v) {
        return v instanceof DenseElementsAttr d && d.numElements() == 0;
    }

    private static boolean isSplatArray(Attribute v, String enumValue) {
        return v instanceof ArrayAttr a
                && a.elements().stream().allMatch(e -> isEnum(e, enumValue));
    }

    /** MLIR treats a dense attribute whose elements are all equal as a splat (not when empty). */
    private static boolean isSplat(Attribute v, long value) {
        if (!(v instanceof DenseElementsAttr d) || d.numElements() == 0) {
            return false;
        }
        boolean isBool = d.type().elementType() instanceof IntegerType it && it.isBoolean();
        for (long i = 0; i < d.numElements(); i++) {
            long e = isBool ? (d.getBoolean(i) ? 1 : 0) : d.getLong(i);
            if (e != value) {
                return false;
            }
        }
        return true;
    }

    private static boolean isDefaultResultAccuracy(Attribute v) {
        if (!(v instanceof DialectAttr d) || !d.hasParams()) {
            return false;
        }
        boolean zeroTolerances =
                d.param("atol").map(a -> ((FloatAttr) a).value() == 0.0).orElse(false)
                        && d.param("rtol").map(a -> ((FloatAttr) a).value() == 0.0).orElse(false)
                        && d.param("ulps").map(a -> ((IntegerAttr) a).value() == 0).orElse(false);
        boolean defaultMode =
                d.param("mode")
                        .map(m -> m instanceof DialectAttr dm && "DEFAULT".equals(dm.rawBody()))
                        .orElse(false);
        return zeroTolerances && defaultMode;
    }
}
