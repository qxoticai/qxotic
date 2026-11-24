package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.ConvDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DotDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.GatherDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.ScatterDimensionNumbers;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FutureType;
import com.qxotic.jota.stablehlo.type.TokenType;
import com.qxotic.jota.stablehlo.type.Type;

/**
 * Custom bytecode encodings of the {@code stablehlo} dialect ({@code
 * stablehlo/dialect/StablehloBytecode.cpp}) for plain MLIR bytecode: the inverse of {@link
 * StablehloDecoder}. Attributes and types without a custom encoding here (builtin ones, CHLO's, the
 * block-scaling and sparsity configs) go to {@link BuiltinEncoder}, which writes the textual form
 * under their own dialect when it has no encoding either, as MLIR's writer does.
 */
final class StablehloEncoder {

    private static final String S = MlirBytecodeWriter.STABLEHLO;
    private static final String B = MlirBytecodeWriter.BUILTIN;

    private StablehloEncoder() {}

    /** Encodes {@code a}; returns the dialect owning the entry. */
    static String attribute(AttrTypeWriter w, Attribute a) {
        if (a instanceof EnumAttr e && e.dialect().equals("stablehlo") && enumAttr(w, e)) {
            return S;
        }
        if (a instanceof DialectAttr d && d.dialect().equals("stablehlo") && dialectAttr(w, d)) {
            return S;
        }
        return BuiltinEncoder.attribute(w, a);
    }

    static String type(AttrTypeWriter w, Type t) {
        switch (t) {
            case TokenType ignored -> w.varInt(StablehloDecoder.TOKEN_TYPE);
            case FutureType f -> {
                w.varInt(StablehloDecoder.FUTURE_TYPE);
                w.list(f.types(), e -> w.type(e, S));
            }
            default -> {
                return BuiltinEncoder.type(w, t);
            }
        }
        return S;
    }

    private static boolean enumAttr(AttrTypeWriter w, EnumAttr e) {
        int code;
        String[] names;
        switch (e.kind()) {
            case "comparison_direction" -> {
                code = StablehloDecoder.COMPARISON_DIRECTION;
                names = VhloDecoder.COMPARISON_DIRECTION;
            }
            case "comparison_type" -> {
                code = StablehloDecoder.COMPARISON_TYPE;
                names = VhloDecoder.COMPARISON_TYPE;
            }
            case "fft_type" -> {
                code = StablehloDecoder.FFT_TYPE;
                names = VhloDecoder.FFT_TYPE;
            }
            case "precision" -> {
                code = StablehloDecoder.PRECISION;
                names = VhloDecoder.PRECISION;
            }
            case "rng_algorithm" -> {
                code = StablehloDecoder.RNG_ALGORITHM;
                names = VhloDecoder.RNG_ALGORITHM;
            }
            case "rng_distribution" -> {
                code = StablehloDecoder.RNG_DISTRIBUTION;
                names = VhloDecoder.RNG_DISTRIBUTION;
            }
            case "transpose" -> {
                code = StablehloDecoder.TRANSPOSE;
                names = VhloDecoder.TRANSPOSE;
            }
            default -> {
                return false;
            }
        }
        int value = java.util.Arrays.asList(names).indexOf(e.value()); // names may hold null
        if (value < 0) {
            return false;
        }
        w.varInt(code);
        w.varInt(value);
        return true;
    }

    /** Writes the custom encoding of {@code d}, or returns false (nothing written). */
    private static boolean dialectAttr(AttrTypeWriter w, DialectAttr d) {
        try {
            switch (d.mnemonic()) {
                case "conv" -> {
                    ConvDimensionNumbers c = ConvDimensionNumbers.from(d);
                    w.varInt(StablehloDecoder.CONV_DIMENSION_NUMBERS);
                    w.signedVarInt(c.inputBatchDimension());
                    w.signedVarInt(c.inputFeatureDimension());
                    w.signedVarInts(c.inputSpatialDimensions());
                    w.signedVarInt(c.kernelInputFeatureDimension());
                    w.signedVarInt(c.kernelOutputFeatureDimension());
                    w.signedVarInts(c.kernelSpatialDimensions());
                    w.signedVarInt(c.outputBatchDimension());
                    w.signedVarInt(c.outputFeatureDimension());
                    w.signedVarInts(c.outputSpatialDimensions());
                }
                case "dot" -> {
                    DotDimensionNumbers n = DotDimensionNumbers.from(d);
                    w.varInt(StablehloDecoder.DOT_DIMENSION_NUMBERS);
                    w.signedVarInts(n.lhsBatchingDimensions());
                    w.signedVarInts(n.rhsBatchingDimensions());
                    w.signedVarInts(n.lhsContractingDimensions());
                    w.signedVarInts(n.rhsContractingDimensions());
                }
                case "gather" -> {
                    GatherDimensionNumbers n = GatherDimensionNumbers.from(d);
                    w.varInt(StablehloDecoder.GATHER_DIMENSION_NUMBERS);
                    w.signedVarInts(n.offsetDims());
                    w.signedVarInts(n.collapsedSliceDims());
                    w.signedVarInts(n.operandBatchingDims());
                    w.signedVarInts(n.startIndicesBatchingDims());
                    w.signedVarInts(n.startIndexMap());
                    w.signedVarInt(n.indexVectorDim());
                }
                case "scatter" -> {
                    ScatterDimensionNumbers n = ScatterDimensionNumbers.from(d);
                    w.varInt(StablehloDecoder.SCATTER_DIMENSION_NUMBERS);
                    w.signedVarInts(n.updateWindowDims());
                    w.signedVarInts(n.insertedWindowDims());
                    w.signedVarInts(n.inputBatchingDims());
                    w.signedVarInts(n.scatterIndicesBatchingDims());
                    w.signedVarInts(n.scatterDimsToOperandDims());
                    w.signedVarInt(n.indexVectorDim());
                }
                case "channel_handle" -> {
                    long handle = scalar(d, "handle");
                    long type = scalar(d, "type");
                    w.varInt(StablehloDecoder.CHANNEL_HANDLE);
                    w.signedVarInt(handle);
                    w.signedVarInt(type);
                }
                case "output_operand_alias" -> {
                    long[] output = longs(d, "output_tuple_indices");
                    long operand = scalar(d, "operand_index");
                    long[] operandTuple = longs(d, "operand_tuple_indices");
                    w.varInt(StablehloDecoder.OUTPUT_OPERAND_ALIAS);
                    w.signedVarInts(output);
                    w.signedVarInt(operand);
                    w.signedVarInts(operandTuple);
                }
                case "dot_algorithm" -> {
                    Type lhs = typeParam(d, "lhs_precision_type");
                    Type rhs = typeParam(d, "rhs_precision_type");
                    Type acc = typeParam(d, "accumulation_type");
                    long lhsCount = scalar(d, "lhs_component_count");
                    long rhsCount = scalar(d, "rhs_component_count");
                    long ops = scalar(d, "num_primitive_operations");
                    if (!(d.param("allow_imprecise_accumulation").orElse(null)
                            instanceof BoolAttr imprecise)) {
                        return false;
                    }
                    w.varInt(StablehloDecoder.DOT_ALGORITHM);
                    w.type(lhs, B);
                    w.type(rhs, B);
                    w.type(acc, B);
                    w.signedVarInt(lhsCount);
                    w.signedVarInt(rhsCount);
                    w.signedVarInt(ops);
                    w.writeByte(imprecise.value() ? 1 : 0); // readBool: one byte
                }
                case "result_accuracy_mode" -> {
                    if (d.hasParams()) {
                        return false;
                    }
                    int mode =
                            java.util.Arrays.asList(VhloDecoder.RESULT_ACCURACY_MODE)
                                    .indexOf(d.rawBody());
                    if (mode < 0) {
                        return false;
                    }
                    w.varInt(StablehloDecoder.RESULT_ACCURACY_MODE);
                    w.varInt(mode);
                }
                case "result_accuracy" -> {
                    long atol = f64(d, "atol");
                    long rtol = f64(d, "rtol");
                    long ulps = scalar(d, "ulps");
                    Attribute mode = d.param("mode").orElse(null);
                    if (mode == null) {
                        return false;
                    }
                    w.varInt(StablehloDecoder.RESULT_ACCURACY);
                    w.apInt(atol, 64);
                    w.apInt(rtol, 64);
                    w.signedVarInt(ulps);
                    w.attribute(mode, S);
                }
                case "bounds" -> {
                    if (d.hasParams()) {
                        return false;
                    }
                    long[] bounds = bounds(d.rawBody());
                    w.varInt(StablehloDecoder.TYPE_EXTENSIONS);
                    w.signedVarInts(bounds);
                }
                case "replica_group_mesh_axes" -> {
                    Attribute mesh = d.param("mesh").orElse(null);
                    if (mesh instanceof SymbolRefAttr s && s.path().size() == 1) {
                        mesh = new StringAttr(s.root()); // a mesh name is written as a string
                    }
                    if (mesh == null || !(d.param("axes").orElse(null) instanceof ArrayAttr axes)) {
                        return false;
                    }
                    w.varInt(StablehloDecoder.REPLICA_GROUP_MESH_AXES);
                    w.attribute(mesh, S);
                    w.list(axes.elements(), e -> w.attribute(e, S));
                }
                case "sub_axis_info" -> {
                    long preSize = scalar(d, "pre_size");
                    long size = scalar(d, "size");
                    w.varInt(StablehloDecoder.SUB_AXIS_INFO);
                    w.signedVarInt(preSize);
                    w.signedVarInt(size);
                }
                case "axis_ref" -> {
                    if (!(d.param("name").orElse(null) instanceof StringAttr name)) {
                        return false;
                    }
                    Attribute sub = d.param("sub_axis_info").orElse(null);
                    w.varInt(StablehloDecoder.AXIS_REF);
                    w.string(name.value());
                    w.writeByte(sub != null ? 1 : 0);
                    if (sub != null) {
                        w.attribute(sub, S);
                    }
                }
                case "mesh_axis" -> {
                    if (!(d.param("name").orElse(null) instanceof StringAttr name)) {
                        return false;
                    }
                    long size = scalar(d, "size");
                    w.varInt(StablehloDecoder.MESH_AXIS);
                    w.string(name.value());
                    w.signedVarInt(size);
                }
                case "mesh" -> {
                    if (!(d.param("axes").orElse(null) instanceof ArrayAttr axes)) {
                        return false;
                    }
                    Attribute ids = d.param("device_ids").orElse(null);
                    w.varInt(StablehloDecoder.MESH);
                    w.attribute(axes, S);
                    w.writeByte(ids != null ? 1 : 0);
                    if (ids != null) {
                        w.attribute(ids, S);
                    }
                }
                default -> {
                    return false;
                }
            }
            return true;
        } catch (IllegalArgumentException | BytecodeException e) {
            // a malformed attribute: no custom encoding, and nothing was written yet when the
            // checks above throw before the first write
            return false;
        }
    }

    private static long scalar(DialectAttr d, String name) {
        if (d.param(name).orElse(null) instanceof IntegerAttr i) {
            return i.value();
        }
        throw new BytecodeException("missing integer " + name + " in " + d);
    }

    private static long[] longs(DialectAttr d, String name) {
        if (d.param(name).orElse(null) instanceof DenseArrayAttr a) {
            return a.values();
        }
        return new long[0]; // absent lists are empty
    }

    private static Type typeParam(DialectAttr d, String name) {
        if (d.param(name).orElse(null) instanceof TypeAttr t) {
            return t.type();
        }
        throw new BytecodeException("missing type " + name + " in " + d);
    }

    private static long f64(DialectAttr d, String name) {
        if (d.param(name).orElse(null) instanceof FloatAttr f) {
            return FloatFormat.F64.encode(f.value());
        }
        throw new BytecodeException("missing float " + name + " in " + d);
    }

    /** {@code 3, ?} to {@code [3, ShapedType::kDynamic]}. */
    private static long[] bounds(String body) {
        if (body.isBlank()) {
            return new long[0];
        }
        String[] parts = body.split(",");
        long[] out = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String p = parts[i].trim();
            out[i] = p.equals("?") ? Long.MIN_VALUE : Long.parseLong(p);
        }
        return out;
    }
}
