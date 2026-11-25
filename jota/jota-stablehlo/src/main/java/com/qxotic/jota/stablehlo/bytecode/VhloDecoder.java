package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
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
import java.util.ArrayList;
import java.util.List;

/**
 * Custom bytecode encodings of the VHLO dialect ({@code stablehlo/dialect/VhloBytecode.cpp}),
 * decoded directly into this model, i.e. into the StableHLO spelling of each attribute and type
 * ({@code vhlo.integer_v1} becomes an {@link IntegerAttr}, {@code vhlo.tensor_v1} a dense elements
 * attribute, {@code vhlo.comparison_direction_v1} {@code #stablehlo<comparison_direction ...>}).
 * The absent-value marker {@code #vhlo.type_v1<!vhlo.none_v1>} becomes {@code TypeAttr(none)}.
 */
final class VhloDecoder {

    private VhloDecoder() {}

    /** The {@code none} type used by VHLO to encode absent optional attributes. */
    static final OpaqueType NONE = new OpaqueType("none");

    static final String[] COMPARISON_DIRECTION = {"EQ", "NE", "GE", "GT", "LE", "LT"};
    static final String[] COMPARISON_TYPE = {
        "NOTYPE", "FLOAT", "TOTALORDER", "SIGNED", "UNSIGNED", "WEAKORDER"
    };
    static final String[] CUSTOM_CALL_API_VERSION = {
        "API_VERSION_UNSPECIFIED",
        "API_VERSION_ORIGINAL",
        "API_VERSION_STATUS_RETURNING",
        "API_VERSION_STATUS_RETURNING_UNIFIED",
        "API_VERSION_TYPED_FFI"
    };
    static final String[] FFT_TYPE = {"FFT", "IFFT", "RFFT", "IRFFT"};
    static final String[] PRECISION = {"DEFAULT", "HIGH", "HIGHEST"};
    static final String[] RNG_ALGORITHM = {"DEFAULT", "THREE_FRY", "PHILOX"};
    static final String[] RNG_DISTRIBUTION = {null, "UNIFORM", "NORMAL"};
    static final String[] TRANSPOSE = {"TRANSPOSE_INVALID", "NO_TRANSPOSE", "TRANSPOSE", "ADJOINT"};
    static final String[] RESULT_ACCURACY_MODE = {"DEFAULT", "HIGHEST", "TOLERANCE"};

    static Attribute attribute(DialectReader r) {
        long code = r.varInt();
        return switch ((int) code) {
            case 1 -> new ArrayAttr(r.list(r::attribute));
            case 2 -> {
                long v = r.varInt();
                if (v != 0 && v != 1) {
                    throw r.error("unsupported boolean value " + v);
                }
                yield BoolAttr.of(v == 1);
            }
            case 3 -> enumAttr(r, "comparison_direction", COMPARISON_DIRECTION);
            case 4 -> enumAttr(r, "comparison_type", COMPARISON_TYPE);
            case 5 -> enumAttr(r, "custom_call_api_version", CUSTOM_CALL_API_VERSION);
            case 6 ->
                    new DictionaryAttr(
                            r.list(() -> MlirBytecodeReader.named(r.attribute(), r.attribute())));
            case 7 -> enumAttr(r, "fft_type", FFT_TYPE);
            case 8 -> {
                Type t = r.type();
                if (!(t instanceof FloatType ft)) {
                    throw r.error("vhlo.float_v1 with non-float type " + t);
                }
                yield new FloatAttr(r.apInt(ft.format().bitWidth()), ft);
            }
            case 9 -> {
                Type t = r.type();
                yield new IntegerAttr(r.apInt(BuiltinDecoder.intWidth(t, r)), t);
            }
            case 10 -> {
                long[] outputTupleIndices = r.signedVarInts();
                long operandIndex = r.signedVarInt();
                long[] operandTupleIndices = r.signedVarInts();
                yield DialectAttr.ofParams(
                        "stablehlo",
                        "output_operand_alias",
                        List.of(
                                new NamedAttribute(
                                        "output_tuple_indices",
                                        DenseArrayAttr.i64(outputTupleIndices)),
                                new NamedAttribute("operand_index", IntegerAttr.i64(operandIndex)),
                                new NamedAttribute(
                                        "operand_tuple_indices",
                                        DenseArrayAttr.i64(operandTupleIndices))));
            }
            case 11 -> enumAttr(r, "precision", PRECISION);
            case 12 -> enumAttr(r, "rng_algorithm", RNG_ALGORITHM);
            case 13 -> enumAttr(r, "rng_distribution", RNG_DISTRIBUTION);
            case 14 -> new StringAttr(r.string());
            case 15 -> tensor(r);
            case 16 -> enumAttr(r, "transpose", TRANSPOSE);
            case 17 -> new TypeAttr(r.type());
            case 18 -> bounds(r.signedVarInts());
            case 19 -> {
                long v = r.varInt();
                yield DialectAttr.ofRaw(
                        "stablehlo", "result_accuracy_mode", name(r, v, RESULT_ACCURACY_MODE));
            }
            case 20 -> {
                long atol = r.apInt(64);
                long rtol = r.apInt(64);
                long ulps = r.signedVarInt();
                Attribute mode = r.attribute();
                yield DialectAttr.ofParams(
                        "stablehlo",
                        "result_accuracy",
                        List.of(
                                new NamedAttribute("atol", new FloatAttr(atol, FloatType.F64)),
                                new NamedAttribute("rtol", new FloatAttr(rtol, FloatType.F64)),
                                new NamedAttribute("ulps", IntegerAttr.i64(ulps)),
                                new NamedAttribute("mode", mode)));
            }
            case 21 -> {
                long preSize = r.signedVarInt();
                long size = r.signedVarInt();
                yield DialectAttr.ofParams(
                        "stablehlo",
                        "sub_axis_info",
                        List.of(
                                new NamedAttribute("pre_size", IntegerAttr.i64(preSize)),
                                new NamedAttribute("size", IntegerAttr.i64(size))));
            }
            case 22 -> {
                Attribute name = r.attribute();
                long hasSubAxisInfo = r.varInt();
                List<NamedAttribute> params = new ArrayList<>();
                params.add(new NamedAttribute("name", name));
                if (hasSubAxisInfo != 0) {
                    params.add(new NamedAttribute("sub_axis_info", r.attribute()));
                }
                yield DialectAttr.ofParams("stablehlo", "axis_ref", params);
            }
            case 23 -> {
                Attribute mesh = r.attribute();
                if (mesh instanceof StringAttr s) { // a mesh name refers to a symbol
                    mesh = com.qxotic.jota.stablehlo.attr.SymbolRefAttr.of(s.value());
                }
                Attribute axes = r.attribute();
                yield DialectAttr.ofParams(
                        "stablehlo",
                        "replica_group_mesh_axes",
                        List.of(
                                new NamedAttribute("mesh", mesh),
                                new NamedAttribute("axes", axes)));
            }
            case 24 -> {
                Attribute name = r.attribute();
                long size = r.signedVarInt();
                yield DialectAttr.ofParams(
                        "stablehlo",
                        "mesh_axis",
                        List.of(
                                new NamedAttribute("name", name),
                                new NamedAttribute("size", IntegerAttr.i64(size))));
            }
            case 25 -> {
                Attribute axes = r.attribute();
                Attribute deviceIds = r.optionalAttribute();
                List<NamedAttribute> params = new ArrayList<>();
                params.add(new NamedAttribute("axes", axes));
                if (deviceIds != null) {
                    params.add(new NamedAttribute("device_ids", deviceIds));
                }
                yield DialectAttr.ofParams("stablehlo", "mesh", params);
            }
            default -> throw r.error("unknown VHLO attribute code " + code);
        };
    }

    private static String name(DialectReader r, long v, String[] names) {
        if (v < 0 || v >= names.length || names[(int) v] == null) {
            throw r.error("invalid enum value " + v);
        }
        return names[(int) v];
    }

    private static EnumAttr enumAttr(DialectReader r, String kind, String[] names) {
        return EnumAttr.stablehlo(kind, name(r, r.varInt(), names));
    }

    /** {@code #stablehlo.bounds<3, ?>}: the text spelling of a type extensions encoding. */
    static DialectAttr bounds(long[] bounds) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < bounds.length; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            sb.append(bounds[i] == Long.MIN_VALUE ? "?" : Long.toString(bounds[i]));
        }
        return DialectAttr.ofRaw("stablehlo", "bounds", sb.toString());
    }

    private static Attribute tensor(DialectReader r) {
        Type t = r.type();
        if (!(t instanceof TensorType tt)) {
            throw r.error("expected a ranked tensor type in vhlo.tensor_v1, got " + t.mlir());
        }
        if (!tt.hasStaticShape()) {
            throw r.error("expected a statically shaped tensor type in vhlo.tensor_v1");
        }
        if (tt.elementType() instanceof QuantizedType) {
            throw r.error("unsupported tensor type in TensorV1Attr: " + tt.mlir());
        }
        try {
            tt.numElements();
        } catch (ArithmeticException e) {
            throw r.error("tensor shape element count overflows in vhlo.tensor_v1");
        }
        return BuiltinDecoder.denseElements(r, tt, r.blob());
    }

    static Type type(DialectReader r) {
        long code = r.varInt();
        return switch ((int) code) {
            case 0 -> IntegerType.I1;
            case 1 -> {
                Type et = r.type();
                if (!(et instanceof FloatType ft)) {
                    throw r.error("complex of non-float type " + et);
                }
                yield new ComplexType(ft);
            }
            case 2 -> FloatType.BF16;
            case 3 -> FloatType.F16;
            case 4 -> FloatType.F32;
            case 5 -> FloatType.F64;
            case 6 -> new FloatType(FloatFormat.F8E4M3FN);
            case 7 -> new FloatType(FloatFormat.F8E5M2);
            case 8 -> new FunctionType(r.list(r::type), r.list(r::type));
            case 9 -> IndexType.INSTANCE;
            case 10 -> IntegerType.signless(4);
            case 11 -> IntegerType.I8;
            case 12 -> IntegerType.I16;
            case 13 -> IntegerType.I32;
            case 14 -> IntegerType.I64;
            case 15 -> IntegerType.unsigned(4);
            case 16 -> IntegerType.unsigned(8);
            case 17 -> IntegerType.unsigned(16);
            case 18 -> IntegerType.unsigned(32);
            case 19 -> IntegerType.unsigned(64);
            case 20 -> rankedTensor(r, null);
            case 21 -> rankedTensor(r, r.attribute());
            case 22 -> TokenType.INSTANCE;
            case 23 -> new TupleType(r.list(r::type));
            case 24 -> uniformQuantized(r);
            case 25 -> new OpaqueType("tensor<*x" + r.type().mlir() + ">");
            case 26 -> new OpaqueType("!shape.witness");
            case 27 -> new FloatType(FloatFormat.F8E4M3FNUZ);
            case 28 -> new FloatType(FloatFormat.F8E5M2FNUZ);
            case 29 -> new FloatType(FloatFormat.F8E4M3B11FNUZ);
            case 30 -> uniformQuantizedPerAxis(r);
            case 31 -> IntegerType.signless(2);
            case 32 -> IntegerType.unsigned(2);
            case 33 -> NONE;
            case 34 -> new FloatType(FloatFormat.TF32);
            case 35 -> new FloatType(FloatFormat.F8E4M3);
            case 36 -> new FloatType(FloatFormat.F8E3M4);
            case 37 -> new FloatType(FloatFormat.F4E2M1FN);
            case 38 -> new FloatType(FloatFormat.F6E2M3FN);
            case 39 -> new FloatType(FloatFormat.F6E3M2FN);
            case 40 -> new FloatType(FloatFormat.F8E8M0FNU);
            case 41 -> { // RankedBufferV1Type: shape + element type, i.e. memref<...>
                long[] shape = r.signedVarInts();
                Type et = r.type();
                StringBuilder sb = new StringBuilder("memref<");
                for (long d : shape) {
                    sb.append(d == Long.MIN_VALUE ? "?" : Long.toString(d)).append('x');
                }
                yield new OpaqueType(sb.append(et.mlir()).append('>').toString());
            }
            case 42 -> new FutureType(r.list(r::type));
            case 43 -> new FloatType(FloatFormat.F8E5M3FNU);
            default -> throw r.error("unsupported VHLO type code " + code);
        };
    }

    private static TensorType rankedTensor(DialectReader r, Attribute encoding) {
        long[] shape = r.signedVarInts();
        Type et = r.type();
        if (!(et instanceof ElementType elementType)) {
            throw r.error("unsupported tensor element type " + et.mlir());
        }
        List<Long> dims = new ArrayList<>(shape.length);
        for (long d : shape) {
            if (d < 0 && d != Long.MIN_VALUE) {
                throw r.error("negative tensor dimension " + d);
            }
            dims.add(d == Long.MIN_VALUE ? TensorType.DYNAMIC : d);
        }
        return new TensorType(dims, elementType, encoding);
    }

    private static IntegerType storageType(DialectReader r, Type t, boolean signed) {
        if (!(t instanceof IntegerType it)) {
            throw r.error("quantized storage type must be an integer type, got " + t);
        }
        return signed ? IntegerType.signless(it.bitWidth()) : IntegerType.unsigned(it.bitWidth());
    }

    private static FloatType expressedType(DialectReader r, Type t) {
        if (!(t instanceof FloatType ft)) {
            throw r.error("quantized expressed type must be a float type, got " + t);
        }
        return ft;
    }

    private static Long nonDefaultMin(IntegerType storage, long min) {
        long defaultMin = storage.isUnsigned() ? 0 : -(1L << (storage.bitWidth() - 1));
        return min == defaultMin ? null : min;
    }

    private static Long nonDefaultMax(IntegerType storage, long max) {
        long defaultMax =
                storage.isUnsigned()
                        ? (1L << storage.bitWidth()) - 1
                        : (1L << (storage.bitWidth() - 1)) - 1;
        return max == defaultMax ? null : max;
    }

    private static QuantizedType uniformQuantized(DialectReader r) {
        long flags = r.varInt();
        IntegerType storage = storageType(r, r.type(), (flags & 1) != 0);
        FloatType expressed = expressedType(r, r.type());
        double scale = Double.longBitsToDouble(r.apInt(64));
        List<Double> scales = List.of(scale);
        long zeroPoint = r.signedVarInt();
        List<Long> zeroPoints = List.of(zeroPoint);
        long min = r.signedVarInt();
        long max = r.signedVarInt();
        Long minOrNull = nonDefaultMin(storage, min);
        Long maxOrNull = nonDefaultMax(storage, max);
        if (minOrNull != null || maxOrNull != null) {
            minOrNull = min;
            maxOrNull = max;
        }
        return new QuantizedType(
                storage, minOrNull, maxOrNull, expressed, null, scales, zeroPoints);
    }

    private static QuantizedType uniformQuantizedPerAxis(DialectReader r) {
        long flags = r.varInt();
        IntegerType storage = storageType(r, r.type(), (flags & 1) != 0);
        FloatType expressed = expressedType(r, r.type());
        long dimension = r.varInt();
        long min = r.signedVarInt();
        long max = r.signedVarInt();
        List<Double> scales = r.list(() -> Double.longBitsToDouble(r.apInt(64)));
        List<Long> zeroPoints = r.list(r::signedVarInt);
        Long minOrNull = nonDefaultMin(storage, min);
        Long maxOrNull = nonDefaultMax(storage, max);
        if (minOrNull != null || maxOrNull != null) {
            minOrNull = min;
            maxOrNull = max;
        }
        return new QuantizedType(
                storage, minOrNull, maxOrNull, expressed, (int) dimension, scales, zeroPoints);
    }
}
