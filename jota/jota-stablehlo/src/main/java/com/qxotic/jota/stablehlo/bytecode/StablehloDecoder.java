package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
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
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FutureType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Custom bytecode encodings of the {@code stablehlo} dialect ({@code
 * stablehlo/dialect/StablehloBytecode.cpp}), as written in plain (non-portable) MLIR bytecode,
 * decoded into the same attributes the text parser builds.
 */
final class StablehloDecoder {

    private StablehloDecoder() {}

    static final int CHANNEL_HANDLE = 1;
    static final int COMPARISON_DIRECTION = 2;
    static final int COMPARISON_TYPE = 3;
    static final int CONV_DIMENSION_NUMBERS = 4;
    static final int DOT_DIMENSION_NUMBERS = 5;
    static final int FFT_TYPE = 6;
    static final int GATHER_DIMENSION_NUMBERS = 7;
    static final int PRECISION = 8;
    static final int RNG_ALGORITHM = 9;
    static final int RNG_DISTRIBUTION = 10;
    static final int SCATTER_DIMENSION_NUMBERS = 11;
    static final int TRANSPOSE = 12;
    static final int TYPE_EXTENSIONS = 13;
    static final int OUTPUT_OPERAND_ALIAS = 14;
    static final int DOT_ALGORITHM = 15;
    static final int RESULT_ACCURACY_MODE = 16;
    static final int RESULT_ACCURACY = 17;
    static final int REPLICA_GROUP_MESH_AXES = 18;
    static final int SUB_AXIS_INFO = 19;
    static final int AXIS_REF = 20;
    static final int MESH_AXIS = 21;
    static final int MESH = 22;
    static final int TENSOR_BLOCK_SCALING_CONFIG = 23;
    static final int BLOCK_SCALING_CONFIG = 24;
    static final int TENSOR_SPARSITY_CONFIG = 25;
    static final int SPARSITY_CONFIG = 26;

    static final int TOKEN_TYPE = 0;
    static final int FUTURE_TYPE = 1;

    static Attribute attribute(DialectReader r) {
        long code = r.varInt();
        return switch ((int) code) {
            case CHANNEL_HANDLE -> {
                long handle = r.signedVarInt();
                long type = r.signedVarInt();
                yield params(
                        "channel_handle",
                        new NamedAttribute("handle", IntegerAttr.i64(handle)),
                        new NamedAttribute("type", IntegerAttr.i64(type)));
            }
            case COMPARISON_DIRECTION ->
                    enumAttr(r, "comparison_direction", VhloDecoder.COMPARISON_DIRECTION);
            case COMPARISON_TYPE -> enumAttr(r, "comparison_type", VhloDecoder.COMPARISON_TYPE);
            case CONV_DIMENSION_NUMBERS ->
                    new ConvDimensionNumbers(
                                    r.signedVarInt(),
                                    r.signedVarInt(),
                                    r.signedVarInts(),
                                    r.signedVarInt(),
                                    r.signedVarInt(),
                                    r.signedVarInts(),
                                    r.signedVarInt(),
                                    r.signedVarInt(),
                                    r.signedVarInts())
                            .toAttribute();
            case DOT_DIMENSION_NUMBERS ->
                    new DotDimensionNumbers(
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts())
                            .toAttribute();
            case FFT_TYPE -> enumAttr(r, "fft_type", VhloDecoder.FFT_TYPE);
            case GATHER_DIMENSION_NUMBERS ->
                    new GatherDimensionNumbers(
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInt())
                            .toAttribute();
            case PRECISION -> enumAttr(r, "precision", VhloDecoder.PRECISION);
            case RNG_ALGORITHM -> enumAttr(r, "rng_algorithm", VhloDecoder.RNG_ALGORITHM);
            case RNG_DISTRIBUTION -> enumAttr(r, "rng_distribution", VhloDecoder.RNG_DISTRIBUTION);
            case SCATTER_DIMENSION_NUMBERS ->
                    new ScatterDimensionNumbers(
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInts(),
                                    r.signedVarInt())
                            .toAttribute();
            case TRANSPOSE -> enumAttr(r, "transpose", VhloDecoder.TRANSPOSE);
            case TYPE_EXTENSIONS -> VhloDecoder.bounds(r.signedVarInts());
            case OUTPUT_OPERAND_ALIAS -> {
                long[] outputTupleIndices = r.signedVarInts();
                long operandIndex = r.signedVarInt();
                long[] operandTupleIndices = r.signedVarInts();
                yield params(
                        "output_operand_alias",
                        new NamedAttribute(
                                "output_tuple_indices", DenseArrayAttr.i64(outputTupleIndices)),
                        new NamedAttribute("operand_index", IntegerAttr.i64(operandIndex)),
                        new NamedAttribute(
                                "operand_tuple_indices", DenseArrayAttr.i64(operandTupleIndices)));
            }
            case DOT_ALGORITHM -> {
                Type lhs = r.type();
                Type rhs = r.type();
                Type accumulation = r.type();
                long lhsCount = r.signedVarInt();
                long rhsCount = r.signedVarInt();
                long primitiveOps = r.signedVarInt();
                boolean imprecise = r.bool();
                yield params(
                        "dot_algorithm",
                        new NamedAttribute("lhs_precision_type", new TypeAttr(lhs)),
                        new NamedAttribute("rhs_precision_type", new TypeAttr(rhs)),
                        new NamedAttribute("accumulation_type", new TypeAttr(accumulation)),
                        new NamedAttribute("lhs_component_count", IntegerAttr.i64(lhsCount)),
                        new NamedAttribute("rhs_component_count", IntegerAttr.i64(rhsCount)),
                        new NamedAttribute(
                                "num_primitive_operations", IntegerAttr.i64(primitiveOps)),
                        new NamedAttribute("allow_imprecise_accumulation", BoolAttr.of(imprecise)));
            }
            case RESULT_ACCURACY_MODE ->
                    DialectAttr.ofRaw(
                            "stablehlo",
                            "result_accuracy_mode",
                            name(r, r.varInt(), VhloDecoder.RESULT_ACCURACY_MODE));
            case RESULT_ACCURACY -> {
                long atol = r.apInt(64);
                long rtol = r.apInt(64);
                long ulps = r.signedVarInt();
                Attribute mode = r.attribute();
                yield params(
                        "result_accuracy",
                        new NamedAttribute("atol", new FloatAttr(atol, FloatType.F64)),
                        new NamedAttribute("rtol", new FloatAttr(rtol, FloatType.F64)),
                        new NamedAttribute("ulps", IntegerAttr.i64(ulps)),
                        new NamedAttribute("mode", mode));
            }
            case REPLICA_GROUP_MESH_AXES -> {
                Attribute mesh = r.attribute();
                if (mesh instanceof StringAttr s) { // a mesh name refers to a symbol
                    mesh = com.qxotic.jota.stablehlo.attr.SymbolRefAttr.of(s.value());
                }
                ArrayAttr axes = new ArrayAttr(r.list(r::attribute));
                yield params(
                        "replica_group_mesh_axes",
                        new NamedAttribute("mesh", mesh),
                        new NamedAttribute("axes", axes));
            }
            case SUB_AXIS_INFO -> {
                long preSize = r.signedVarInt();
                long size = r.signedVarInt();
                yield params(
                        "sub_axis_info",
                        new NamedAttribute("pre_size", IntegerAttr.i64(preSize)),
                        new NamedAttribute("size", IntegerAttr.i64(size)));
            }
            case AXIS_REF -> {
                List<NamedAttribute> p = new ArrayList<>();
                p.add(new NamedAttribute("name", new StringAttr(r.string())));
                if (r.bool()) {
                    p.add(new NamedAttribute("sub_axis_info", r.attribute()));
                }
                yield DialectAttr.ofParams("stablehlo", "axis_ref", p);
            }
            case MESH_AXIS -> {
                String axisName = r.string();
                long size = r.signedVarInt();
                yield params(
                        "mesh_axis",
                        new NamedAttribute("name", new StringAttr(axisName)),
                        new NamedAttribute("size", IntegerAttr.i64(size)));
            }
            case MESH -> {
                List<NamedAttribute> p = new ArrayList<>();
                p.add(new NamedAttribute("axes", r.attribute()));
                if (r.bool()) {
                    p.add(new NamedAttribute("device_ids", r.attribute()));
                }
                yield DialectAttr.ofParams("stablehlo", "mesh", p);
            }
            case TENSOR_BLOCK_SCALING_CONFIG -> {
                List<NamedAttribute> f = new ArrayList<>();
                f.add(new NamedAttribute("scale_idx", IntegerAttr.i64(r.signedVarInt())));
                if (r.bool()) {
                    f.add(new NamedAttribute("zero_idx", IntegerAttr.i64(r.signedVarInt())));
                }
                for (String list : List.of("strides", "steps", "block_sizes")) {
                    long[] values = r.signedVarInts();
                    if (values.length > 0) {
                        f.add(new NamedAttribute(list, DenseArrayAttr.i64(values)));
                    }
                }
                yield new DictionaryAttr(f);
            }
            case BLOCK_SCALING_CONFIG -> perOperand(r, "block_scaling_config");
            case TENSOR_SPARSITY_CONFIG -> {
                List<NamedAttribute> f = new ArrayList<>();
                for (String field :
                        List.of("num_non_zero", "block_size", "dimension", "stride", "idx")) {
                    f.add(new NamedAttribute(field, IntegerAttr.i64(r.signedVarInt())));
                }
                yield new DictionaryAttr(f);
            }
            case SPARSITY_CONFIG -> perOperand(r, "sparsity_config");
            default -> throw r.error("unknown stablehlo attribute code " + code);
        };
    }

    static Type type(DialectReader r) {
        long code = r.varInt();
        return switch ((int) code) {
            case TOKEN_TYPE -> TokenType.INSTANCE;
            case FUTURE_TYPE -> new FutureType(r.list(r::type));
            default -> throw r.error("unknown stablehlo type code " + code);
        };
    }

    /** {@code #stablehlo.<mnemonic><lhs = <...>, rhs = <...>>}, each side optional. */
    private static DialectAttr perOperand(DialectReader r, String mnemonic) {
        List<NamedAttribute> sides = new ArrayList<>();
        for (String side : List.of("lhs", "rhs")) {
            Attribute a = r.optionalAttribute();
            if (a != null) {
                sides.add(new NamedAttribute(side, a));
            }
        }
        return DialectAttr.ofParams("stablehlo", mnemonic, sides);
    }

    private static DialectAttr params(String mnemonic, NamedAttribute... params) {
        return DialectAttr.ofParams("stablehlo", mnemonic, List.of(params));
    }

    private static EnumAttr enumAttr(DialectReader r, String kind, String[] names) {
        return EnumAttr.stablehlo(kind, name(r, r.varInt(), names));
    }

    private static String name(DialectReader r, long v, String[] names) {
        if (v < 0 || v >= names.length || names[(int) v] == null) {
            throw r.error("invalid enum value " + v);
        }
        return names[(int) v];
    }
}
