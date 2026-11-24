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
import com.qxotic.jota.stablehlo.attr.OpaqueAttr;
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
import com.qxotic.jota.stablehlo.portable.StableHloVersion;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import com.qxotic.jota.stablehlo.type.TensorType;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Legalizes a StableHLO program to VHLO for a target version, following {@code
 * stablehlo/transforms/StablehloLegalizeToVhlo.cpp} and {@code VhloToVersionPatterns.td}.
 *
 * <p>Per op: default values are added for every attribute VHLO requires ({@code addDefaults}),
 * dimension-number structs are exploded into flat attributes, dense arrays become tensors and
 * symbols become strings ({@code convertSpecial} / {@code convertGeneric}). The op version is the
 * one whose version window contains the target; attributes that the chosen (older) version does not
 * have are dropped when they hold their default and rejected otherwise, as the downgrade patterns
 * do.
 *
 * <p>The output keeps the decoded attribute spelling of {@link
 * com.qxotic.jota.stablehlo.bytecode.MlirBytecodeReader} (e.g. {@code #stablehlo<precision
 * DEFAULT>} for a {@code vhlo.precision_v1}), which the bytecode writer encodes as VHLO.
 */
public final class StableHloToVhlo {

    private static final TypeAttr NONE = new TypeAttr(new OpaqueType("none"));

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

    /** Ops whose first VHLO version takes a single operand (variadic since v2). */
    private static final Set<String> SINGLE_OPERAND_V1 =
            Set.of("all_gather", "all_reduce", "all_to_all", "collective_broadcast");

    private static final List<String> API_VERSIONS =
            List.of(
                    "API_VERSION_UNSPECIFIED",
                    "API_VERSION_ORIGINAL",
                    "API_VERSION_STATUS_RETURNING",
                    "API_VERSION_STATUS_RETURNING_UNIFIED",
                    "API_VERSION_TYPED_FFI");

    private final StableHloVersion target;
    private final Map<Value, Value> values = new IdentityHashMap<>();

    private StableHloToVhlo(StableHloVersion target) {
        this.target = target;
    }

    /** Converts a StableHLO program to VHLO ops valid at {@code target}. */
    public static Program convert(Program program, StableHloVersion target) {
        if (!target.isSupported()) {
            throw new VhloException(
                    "target version "
                            + target
                            + " outside ["
                            + StableHloVersion.MINIMUM
                            + ", "
                            + StableHloVersion.CURRENT
                            + "]");
        }
        return new Program(new StableHloToVhlo(target).convert(program.operation()));
    }

    private Operation convert(Operation op) {
        String name;
        List<NamedAttribute> inherent;
        List<NamedAttribute> discardable;
        if (op.name().equals("builtin.module")) {
            name = op.name();
            inherent = op.properties();
            discardable = op.attributes();
        } else {
            String base = baseName(op);
            VhloOpSchemas.Schema schema = schemaFor(base, op.name());
            name = "vhlo." + schema.mnemonic();
            Map<String, Attribute> attrs = new LinkedHashMap<>();
            for (NamedAttribute a : op.allAttributes()) {
                convertAttribute(base, a.name(), a.value(), attrs);
            }
            addDefaults(base, op, attrs);
            if (base.equals("compare")
                    && String.valueOf(attrs.get("compare_type")).contains("WEAKORDER")
                    && target.compareTo(new StableHloVersion(1, 22, 0)) < 0) {
                throw new VhloException(
                        "comparison type WEAKORDER is not available in StableHLO " + target);
            }
            downgrade(base, schema, op, attrs);
            inherent = new ArrayList<>();
            discardable = new ArrayList<>();
            for (String attr : schema.attributes()) {
                Attribute v = attrs.remove(attr);
                if (v == null) {
                    throw new VhloException(op.name() + ": missing attribute '" + attr + "'");
                }
                inherent.add(new NamedAttribute(attr, v));
            }
            for (Map.Entry<String, Attribute> e : attrs.entrySet()) {
                discardable.add(new NamedAttribute(e.getKey(), e.getValue()));
            }
        }
        List<Value> operands = new ArrayList<>();
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
        for (int i = 0; i < op.regions().size(); i++) {
            for (Block b : op.region(i).blocks()) {
                Block nb = new Block();
                for (Value arg : b.arguments()) {
                    values.put(arg, nb.addArgument(arg.type(), null));
                }
                for (Operation inner : b.operations()) {
                    nb.addOperation(convert(inner));
                }
                regions.get(i).addBlock(nb);
            }
        }
        return out;
    }

    private static String baseName(Operation op) {
        return switch (op.name()) {
            case "func.func" -> "func";
            case "func.call" -> "call";
            case "func.return", "stablehlo.return" -> "return";
            default -> {
                if (!op.dialect().equals("stablehlo")) {
                    throw new VhloException("no VHLO form for operation " + op.name());
                }
                yield op.mnemonic();
            }
        };
    }

    /** The version of {@code base} whose window contains the target. */
    private VhloOpSchemas.Schema schemaFor(String base, String opName) {
        for (int v = 1; v <= 9; v++) {
            VhloOpSchemas.Schema s = VhloOpSchemas.get(base + "_v" + v);
            if (s == null) {
                continue;
            }
            StableHloVersion min = StableHloVersion.parse(s.minVersion());
            StableHloVersion max =
                    s.maxVersion().equals("current")
                            ? StableHloVersion.CURRENT
                            : StableHloVersion.parse(s.maxVersion());
            if (target.compareTo(min) >= 0 && target.compareTo(max) <= 0) {
                return s;
            }
        }
        throw new VhloException(opName + " is not available in StableHLO " + target);
    }

    // ================================================================== convertSpecial

    private static void convertAttribute(
            String op, String name, Attribute value, Map<String, Attribute> out) {
        if (COLLECTIVES.contains(op) && name.equals("channel_handle")) {
            out.put("channel_id", IntegerAttr.i64(param(value, "handle")));
            return;
        }
        if (COLLECTIVES.contains(op)
                && (name.equals("use_global_device_ids") || name.equals("has_dynamic_root"))) {
            out.put(name, BoolAttr.TRUE);
            return;
        }
        switch (op + "." + name) {
            case "convolution.dimension_numbers", "dynamic_conv.dimension_numbers" -> {
                ConvDimensionNumbers c = ConvDimensionNumbers.from(value);
                out.put("input_batch_dimension", IntegerAttr.i64(c.inputBatchDimension()));
                out.put("input_feature_dimension", IntegerAttr.i64(c.inputFeatureDimension()));
                out.put("input_spatial_dimensions", tensor(c.inputSpatialDimensions()));
                out.put(
                        "kernel_input_feature_dimension",
                        IntegerAttr.i64(c.kernelInputFeatureDimension()));
                out.put(
                        "kernel_output_feature_dimension",
                        IntegerAttr.i64(c.kernelOutputFeatureDimension()));
                out.put("kernel_spatial_dimensions", tensor(c.kernelSpatialDimensions()));
                out.put("output_batch_dimension", IntegerAttr.i64(c.outputBatchDimension()));
                out.put("output_feature_dimension", IntegerAttr.i64(c.outputFeatureDimension()));
                out.put("output_spatial_dimensions", tensor(c.outputSpatialDimensions()));
            }
            case "custom_call.api_version" -> {
                int v = (int) ((IntegerAttr) value).value();
                out.put(name, EnumAttr.stablehlo("custom_call_api_version", API_VERSIONS.get(v)));
            }
            case "custom_call.called_computations" -> {
                List<Attribute> names = new ArrayList<>();
                for (Attribute s : ((ArrayAttr) value).elements()) {
                    names.add(new StringAttr(((SymbolRefAttr) s).leaf()));
                }
                out.put(name, new ArrayAttr(names));
            }
            case "dot_general.dot_dimension_numbers" -> {
                DotDimensionNumbers d = DotDimensionNumbers.from(value);
                out.put("lhs_batching_dimensions", tensor(d.lhsBatchingDimensions()));
                out.put("rhs_batching_dimensions", tensor(d.rhsBatchingDimensions()));
                out.put("lhs_contracting_dimensions", tensor(d.lhsContractingDimensions()));
                out.put("rhs_contracting_dimensions", tensor(d.rhsContractingDimensions()));
            }
            case "dot_general.algorithm" -> {
                DialectAttr d = (DialectAttr) value;
                for (String f :
                        List.of(
                                "lhs_precision_type",
                                "rhs_precision_type",
                                "accumulation_type",
                                "lhs_component_count",
                                "rhs_component_count",
                                "num_primitive_operations",
                                "allow_imprecise_accumulation")) {
                    Attribute v =
                            d.param(f)
                                    .orElseThrow(() -> new VhloException("algorithm without " + f));
                    out.put(f, v instanceof IntegerAttr i ? IntegerAttr.i64(i.value()) : v);
                }
            }
            case "dot_general.block_scaling_config", "dot_general.sparsity_config" ->
                    out.put(name, perOperandConfig((DialectAttr) value));
            case "gather.dimension_numbers", "dynamic_gather.dimension_numbers" -> {
                GatherDimensionNumbers g = GatherDimensionNumbers.from(value);
                out.put("offset_dims", tensor(g.offsetDims()));
                out.put("collapsed_slice_dims", tensor(g.collapsedSliceDims()));
                out.put("operand_batching_dims", tensor(g.operandBatchingDims()));
                out.put("start_indices_batching_dims", tensor(g.startIndicesBatchingDims()));
                out.put("start_index_map", tensor(g.startIndexMap()));
                out.put("index_vector_dim", IntegerAttr.i64(g.indexVectorDim()));
            }
            case "scatter.scatter_dimension_numbers" -> {
                ScatterDimensionNumbers s = ScatterDimensionNumbers.from(value);
                out.put("update_window_dims", tensor(s.updateWindowDims()));
                out.put("inserted_window_dims", tensor(s.insertedWindowDims()));
                out.put("input_batching_dims", tensor(s.inputBatchingDims()));
                out.put("scatter_indices_batching_dims", tensor(s.scatterIndicesBatchingDims()));
                out.put("scatter_dims_to_operand_dims", tensor(s.scatterDimsToOperandDims()));
                out.put("index_vector_dim", IntegerAttr.i64(s.indexVectorDim()));
            }
            case "send.channel_handle", "recv.channel_handle" -> {
                out.put("channel_id", IntegerAttr.i64(param(value, "handle")));
                out.put("channel_type", IntegerAttr.i64(param(value, "type")));
            }
            default -> out.put(name, generic(value));
        }
    }

    /** Port of {@code convertGeneric}: the VHLO shape of a StableHLO attribute. */
    static Attribute generic(Attribute a) {
        return switch (a) {
            case DenseArrayAttr d when d.elementType() instanceof IntegerType -> toTensor(d);
            case DenseArrayAttr d -> throw new VhloException("no VHLO form for " + d);
            case SymbolRefAttr s -> new StringAttr(s.root());
            case ArrayAttr arr ->
                    new ArrayAttr(arr.elements().stream().map(StableHloToVhlo::generic).toList());
            case DictionaryAttr d -> {
                List<NamedAttribute> entries = new ArrayList<>();
                for (NamedAttribute e : d.entries()) {
                    entries.add(new NamedAttribute(e.name(), generic(e.value())));
                }
                yield new DictionaryAttr(entries);
            }
            case IntegerAttr i when i.type() instanceof IntegerType it && it.isBoolean() ->
                    BoolAttr.of(i.value() != 0);
            case UnitAttr u -> throw new VhloException("unit attributes have no VHLO form");
            case OpaqueAttr o -> throw new VhloException("no VHLO form for " + o);
            case DialectAttr d when d.dialect().equals("stablehlo") && d.hasParams() ->
                    DialectAttr.ofParams(
                            d.dialect(),
                            d.mnemonic(),
                            d.params().stream()
                                    .map(
                                            p ->
                                                    new NamedAttribute(
                                                            p.name(),
                                                            p.value() instanceof SymbolRefAttr s
                                                                    ? new StringAttr(s.root())
                                                                    : p.value()))
                                    .toList());
            default -> a;
        };
    }

    /** {@code #stablehlo.block_scaling_config<lhs = <...>>} → {@code {lhs = {...}}}. */
    private static DictionaryAttr perOperandConfig(DialectAttr d) {
        List<NamedAttribute> sides = new ArrayList<>();
        for (NamedAttribute side : d.params()) {
            List<NamedAttribute> fields = new ArrayList<>();
            for (NamedAttribute f : ((DictionaryAttr) side.value()).entries()) {
                Attribute v = f.value();
                if (v instanceof DenseArrayAttr arr) {
                    List<Attribute> ints = new ArrayList<>();
                    for (long x : arr.values()) {
                        ints.add(IntegerAttr.i64(x));
                    }
                    v = new ArrayAttr(ints);
                }
                fields.add(new NamedAttribute(f.name(), v));
            }
            sides.add(new NamedAttribute(side.name(), new DictionaryAttr(fields)));
        }
        return new DictionaryAttr(sides);
    }

    // ================================================================== addDefaults

    private static void addDefaults(String op, Operation source, Map<String, Attribute> a) {
        switch (op) {
            case "func" -> {
                a.putIfAbsent("sym_visibility", new StringAttr(""));
                a.putIfAbsent("arg_attrs", new ArrayAttr(List.of()));
                a.putIfAbsent("res_attrs", new ArrayAttr(List.of()));
            }
            case "cholesky" -> a.putIfAbsent("lower", BoolAttr.FALSE);
            case "compare" ->
                    a.putIfAbsent("compare_type", EnumAttr.stablehlo("comparison_type", "NOTYPE"));
            case "composite" -> {
                a.putIfAbsent("version", IntegerAttr.i64(0));
                a.putIfAbsent("composite_attributes", new DictionaryAttr(List.of()));
            }
            case "convolution", "dynamic_conv" -> {
                int n = spatialDims(a);
                a.putIfAbsent("window_strides", tensor(filled(n, 1)));
                if (op.equals("convolution")) {
                    a.putIfAbsent("padding", padding(n));
                }
                a.putIfAbsent("lhs_dilation", tensor(filled(n, 1)));
                a.putIfAbsent("rhs_dilation", tensor(filled(n, 1)));
                a.putIfAbsent("window_reversal", booleans(n));
            }
            case "custom_call" -> {
                a.putIfAbsent("has_side_effect", BoolAttr.FALSE);
                boolean typedFfi =
                        a.get("api_version") instanceof EnumAttr e
                                && e.value().equals("API_VERSION_TYPED_FFI");
                a.putIfAbsent(
                        "backend_config",
                        typedFfi ? new DictionaryAttr(List.of()) : new StringAttr(""));
                a.putIfAbsent(
                        "api_version",
                        EnumAttr.stablehlo("custom_call_api_version", "API_VERSION_ORIGINAL"));
                a.putIfAbsent("called_computations", new ArrayAttr(List.of()));
                a.putIfAbsent("operand_layouts", new ArrayAttr(List.of()));
                a.putIfAbsent("result_layouts", new ArrayAttr(List.of()));
                a.putIfAbsent("output_operand_aliases", new ArrayAttr(List.of()));
                a.putIfAbsent("result_tilings", new ArrayAttr(List.of()));
            }
            case "dot_general" -> {
                for (String f :
                        List.of(
                                "lhs_precision_type",
                                "rhs_precision_type",
                                "accumulation_type",
                                "lhs_component_count",
                                "rhs_component_count",
                                "num_primitive_operations",
                                "allow_imprecise_accumulation",
                                "block_scaling_config",
                                "sparsity_config")) {
                    a.putIfAbsent(f, NONE);
                }
            }
            case "dynamic_broadcast_in_dim" -> {
                a.putIfAbsent("known_expanding_dimensions", tensor(new long[0]));
                a.putIfAbsent("known_nonexpanding_dimensions", tensor(new long[0]));
            }
            case "infeed" -> {
                a.putIfAbsent("infeed_config", new StringAttr(""));
                a.putIfAbsent("layout", new ArrayAttr(List.of()));
            }
            case "outfeed" -> a.putIfAbsent("outfeed_config", new StringAttr(""));
            case "send", "recv" -> {
                a.putIfAbsent("is_host_transfer", BoolAttr.FALSE);
                a.putIfAbsent("source_target_pairs", tensor(new long[0]));
            }
            case "reduce_window", "select_and_scatter" -> {
                Attribute dims = a.get("window_dimensions");
                if (dims == null) {
                    throw new VhloException("stablehlo." + op + " needs window_dimensions");
                }
                int n = (int) ((DenseElementsAttr) dims).numElements();
                a.putIfAbsent("window_strides", tensor(filled(n, 1)));
                if (op.equals("reduce_window")) {
                    a.putIfAbsent("base_dilations", tensor(filled(n, 1)));
                    a.putIfAbsent("window_dilations", tensor(filled(n, 1)));
                }
                a.putIfAbsent("padding", padding(n));
            }
            case "scatter" -> a.putIfAbsent("unique_indices", BoolAttr.FALSE);
            case "sort" -> {
                a.putIfAbsent("dimension", IntegerAttr.i64(-1));
                a.putIfAbsent("is_stable", BoolAttr.FALSE);
            }
            default -> {}
        }
        if (COLLECTIVES.contains(op)) {
            a.putIfAbsent("channel_id", IntegerAttr.i64(0));
            if (Set.of("all_gather", "all_reduce", "collective_reduce", "reduce_scatter")
                    .contains(op)) {
                a.putIfAbsent("use_global_device_ids", BoolAttr.FALSE);
            }
            if (Set.of("collective_broadcast", "collective_reduce").contains(op)) {
                a.putIfAbsent("has_dynamic_root", BoolAttr.FALSE);
            }
        }
        if (Set.of("convolution", "dynamic_conv", "dot_general", "dot").contains(op)) {
            a.putIfAbsent(
                    "precision_config",
                    new ArrayAttr(
                            List.of(
                                    EnumAttr.stablehlo("precision", "DEFAULT"),
                                    EnumAttr.stablehlo("precision", "DEFAULT"))));
        }
        if (Set.of("gather", "dynamic_gather", "scatter").contains(op)) {
            a.putIfAbsent("indices_are_sorted", BoolAttr.FALSE);
        }
        if (RESULT_ACCURACY_OPS.contains(op)) {
            a.putIfAbsent("result_accuracy", defaultResultAccuracy());
        }
    }

    static DialectAttr defaultResultAccuracy() {
        return DialectAttr.ofParams(
                "stablehlo",
                "result_accuracy",
                List.of(
                        new NamedAttribute("atol", FloatAttr.of(0.0, FloatType.F64)),
                        new NamedAttribute("rtol", FloatAttr.of(0.0, FloatType.F64)),
                        new NamedAttribute("ulps", IntegerAttr.i64(0)),
                        new NamedAttribute(
                                "mode",
                                DialectAttr.ofRaw(
                                        "stablehlo", "result_accuracy_mode", "DEFAULT"))));
    }

    // ================================================================== downgrade

    /**
     * Fits the attributes of the latest op version into {@code schema} (possibly an older version):
     * attributes it does not know must hold their default value.
     */
    private void downgrade(
            String op, VhloOpSchemas.Schema schema, Operation source, Map<String, Attribute> a) {
        Set<String> known = Set.copyOf(schema.attributes());
        if (schema.mnemonic().endsWith("_v1")
                && SINGLE_OPERAND_V1.contains(op)
                && source.numOperands() != 1) {
            throw new VhloException(
                    "stablehlo."
                            + op
                            + " with "
                            + source.numOperands()
                            + " operands needs StableHLO >= the version of its _v2 (target "
                            + target
                            + ")");
        }
        for (String name : List.copyOf(a.keySet())) {
            if (known.contains(name) || isDiscardable(name, schema)) {
                continue;
            }
            Attribute v = a.get(name);
            if (!isDowngradeDefault(name, v)) {
                throw new VhloException(
                        "stablehlo."
                                + op
                                + ": attribute '"
                                + name
                                + "' = "
                                + v
                                + " cannot be expressed in StableHLO "
                                + target);
            }
            a.remove(name);
        }
        if (schema.mnemonic().equals("dynamic_conv_v1") && !a.containsKey("padding")) {
            a.put("padding", padding(spatialDims(a)));
        }
    }

    /** Names that are not attributes of any version of the op are discardable attributes. */
    private static boolean isDiscardable(String name, VhloOpSchemas.Schema schema) {
        String base = schema.mnemonic().replaceAll("_v\\d+$", "");
        for (int v = 1; v <= 9; v++) {
            VhloOpSchemas.Schema s = VhloOpSchemas.get(base + "_v" + v);
            if (s != null && s.attributes().contains(name)) {
                return false;
            }
        }
        return true;
    }

    private static boolean isDowngradeDefault(String name, Attribute v) {
        return v.equals(NONE)
                || v.equals(BoolAttr.FALSE)
                || (v instanceof DenseElementsAttr d && d.numElements() == 0)
                || (v instanceof ArrayAttr arr && arr.elements().isEmpty())
                || (name.equals("result_accuracy") && v.equals(defaultResultAccuracy()));
    }

    // ================================================================== helpers

    private static long param(Attribute v, String name) {
        if (v instanceof DialectAttr d && d.param(name).orElse(null) instanceof IntegerAttr i) {
            return i.value();
        }
        throw new VhloException("expected " + name + " in " + v);
    }

    private static int spatialDims(Map<String, Attribute> a) {
        Attribute dims = a.get("input_spatial_dimensions");
        return dims == null ? 0 : (int) ((DenseElementsAttr) dims).numElements();
    }

    private static long[] filled(int n, long value) {
        long[] out = new long[n];
        Arrays.fill(out, value);
        return out;
    }

    static DenseElementsAttr tensor(long[] values) {
        return toTensor(DenseArrayAttr.i64(values));
    }

    private static DenseElementsAttr toTensor(DenseArrayAttr d) {
        IntegerType et = (IntegerType) d.elementType();
        int bytes = et.storageBytes();
        byte[] raw = new byte[d.size() * bytes];
        for (int i = 0; i < d.size(); i++) {
            for (int k = 0; k < bytes; k++) {
                raw[i * bytes + k] = (byte) (d.get(i) >>> (8 * k));
            }
        }
        return new DenseElementsAttr(TensorType.of(et, d.size()), raw, false);
    }

    private static DenseElementsAttr booleans(int n) {
        return new DenseElementsAttr(TensorType.of(IntegerType.I1, n), new byte[n], false);
    }

    private static DenseElementsAttr padding(int n) {
        return new DenseElementsAttr(
                TensorType.of(IntegerType.I64, n, 2), new byte[n * 2 * 8], false);
    }
}
