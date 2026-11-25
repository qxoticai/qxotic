package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DialectAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.EnumAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.type.ComplexType;
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
import java.util.Arrays;

/**
 * VHLO encodings (inverse of {@link VhloDecoder}, {@code stablehlo/dialect/VhloBytecode.cpp}).
 * Input attributes must already have their VHLO shape (see {@link
 * com.qxotic.jota.stablehlo.vhlo.StableHloToVhlo}): dense arrays as tensors, symbols as strings,
 * dimension numbers exploded.
 */
final class VhloEncoder {

    private static final String V = MlirBytecodeWriter.VHLO;

    private VhloEncoder() {}

    static String attribute(AttrTypeWriter w, Attribute a) {
        switch (a) {
            case ArrayAttr arr -> {
                w.varInt(1);
                w.list(arr.elements(), e -> w.attribute(e, V));
            }
            case BoolAttr b -> {
                w.varInt(2);
                w.varInt(b.value() ? 1 : 0);
            }
            case IntegerAttr i when i.type() instanceof IntegerType it && it.isBoolean() -> {
                w.varInt(2);
                w.varInt(i.value() != 0 ? 1 : 0);
            }
            case EnumAttr e -> enumAttr(w, e);
            case DictionaryAttr d -> {
                w.varInt(6);
                w.list(
                        d.entries(),
                        (NamedAttribute e) -> {
                            w.attribute(new StringAttr(e.name()), V);
                            w.attribute(e.value(), V);
                        });
            }
            case FloatAttr f -> {
                w.varInt(8);
                w.type(f.type(), V);
                w.apInt(f.bits(), f.type().format().bitWidth());
            }
            case IntegerAttr i -> {
                w.varInt(9);
                w.type(i.type(), V);
                w.apInt(i.value(), i.type() instanceof IntegerType it ? it.bitWidth() : 64);
            }
            case StringAttr s -> {
                w.varInt(14);
                w.string(s.value());
            }
            case SymbolRefAttr s -> {
                w.varInt(14);
                w.string(s.leaf());
            }
            case DenseElementsAttr d -> {
                if (d.type().elementType() instanceof QuantizedType) {
                    throw new BytecodeException(
                            "unsupported tensor type in TensorV1Attr: " + d.type());
                }
                w.varInt(15);
                w.type(d.type(), V);
                w.blob(BuiltinEncoder.denseBlob(d));
            }
            case DenseArrayAttr d -> // StableHLO dense arrays are tensors in VHLO
                    attribute(w, toTensor(d));
            case TypeAttr t -> {
                w.varInt(17);
                w.type(t.type(), V);
            }
            case DialectAttr d -> dialectAttr(w, d);
            default -> throw new BytecodeException("no VHLO encoding for attribute " + a);
        }
        return V;
    }

    static DenseElementsAttr toTensor(DenseArrayAttr d) {
        IntegerType et = (IntegerType) d.elementType();
        TensorType tt = TensorType.of(et, d.size());
        int bytes = et.storageBytes();
        byte[] raw = new byte[d.size() * bytes];
        for (int i = 0; i < d.size(); i++) {
            for (int k = 0; k < bytes; k++) {
                raw[i * bytes + k] = (byte) (d.get(i) >>> (8 * k));
            }
        }
        return new DenseElementsAttr(tt, raw, false);
    }

    private static void enumAttr(AttrTypeWriter w, EnumAttr e) {
        String[] names;
        int code;
        switch (e.kind()) {
            case "comparison_direction" -> {
                code = 3;
                names = VhloDecoder.COMPARISON_DIRECTION;
            }
            case "comparison_type" -> {
                code = 4;
                names = VhloDecoder.COMPARISON_TYPE;
            }
            case "custom_call_api_version" -> {
                code = 5;
                names = VhloDecoder.CUSTOM_CALL_API_VERSION;
            }
            case "fft_type" -> {
                code = 7;
                names = VhloDecoder.FFT_TYPE;
            }
            case "precision" -> {
                code = 11;
                names = VhloDecoder.PRECISION;
            }
            case "rng_algorithm" -> {
                code = 12;
                names = VhloDecoder.RNG_ALGORITHM;
            }
            case "rng_distribution" -> {
                code = 13;
                names = VhloDecoder.RNG_DISTRIBUTION;
            }
            case "transpose" -> {
                code = 16;
                names = VhloDecoder.TRANSPOSE;
            }
            case "result_accuracy_mode" -> {
                code = 19;
                names = VhloDecoder.RESULT_ACCURACY_MODE;
            }
            default -> throw new BytecodeException("no VHLO encoding for enum " + e);
        }
        int ordinal = Arrays.asList(names).indexOf(e.value());
        if (ordinal < 0) {
            throw new BytecodeException("unknown value of " + e.kind() + ": " + e.value());
        }
        w.varInt(code);
        w.varInt(ordinal);
    }

    private static void dialectAttr(AttrTypeWriter w, DialectAttr d) {
        switch (d.mnemonic()) {
            case "output_operand_alias" -> {
                w.varInt(10);
                w.signedVarInts(longs(d, "output_tuple_indices"));
                w.signedVarInt(longs(d, "operand_index")[0]);
                w.signedVarInts(longs(d, "operand_tuple_indices"));
            }
            case "bounds" -> {
                w.varInt(18);
                w.signedVarInts(bounds(d.rawBody()));
            }
            case "result_accuracy_mode" -> {
                w.varInt(19);
                w.varInt(
                        Arrays.asList(VhloDecoder.RESULT_ACCURACY_MODE)
                                .indexOf(d.rawBody().trim()));
            }
            case "result_accuracy" -> {
                w.varInt(20);
                w.apInt(floatBits(d, "atol"), 64);
                w.apInt(floatBits(d, "rtol"), 64);
                w.signedVarInt(longs(d, "ulps")[0]);
                w.attribute(d.param("mode").orElseThrow(), V);
            }
            case "sub_axis_info" -> {
                w.varInt(21);
                w.signedVarInt(longs(d, "pre_size")[0]);
                w.signedVarInt(longs(d, "size")[0]);
            }
            case "axis_ref" -> {
                w.varInt(22);
                w.attribute(d.param("name").orElseThrow(), V);
                Attribute sub = d.param("sub_axis_info").orElse(null);
                w.varInt(sub == null ? 0 : 1);
                if (sub != null) {
                    w.attribute(sub, V);
                }
            }
            case "replica_group_mesh_axes" -> {
                w.varInt(23);
                w.attribute(d.param("mesh").orElseThrow(), V);
                w.attribute(d.param("axes").orElseThrow(), V);
            }
            case "mesh_axis" -> {
                w.varInt(24);
                w.attribute(d.param("name").orElseThrow(), V);
                w.signedVarInt(longs(d, "size")[0]);
            }
            case "mesh" -> {
                w.varInt(25);
                w.attribute(d.param("axes").orElseThrow(), V);
                Attribute ids = d.param("device_ids").orElse(null);
                w.optionalAttribute(ids, V);
            }
            default -> throw new BytecodeException("no VHLO encoding for attribute " + d);
        }
    }

    private static long floatBits(DialectAttr d, String name) {
        Attribute a = d.param(name).orElseThrow();
        if (a instanceof FloatAttr f) {
            return FloatFormat.F64.encode(f.value());
        }
        throw new BytecodeException("expected a float for " + name + " in " + d);
    }

    private static long[] longs(DialectAttr d, String name) {
        Attribute a =
                d.param(name)
                        .orElseThrow(() -> new BytecodeException("missing " + name + " in " + d));
        return switch (a) {
            case IntegerAttr i -> new long[] {i.value()};
            case DenseArrayAttr arr -> arr.values();
            case ArrayAttr arr ->
                    arr.elements().stream().mapToLong(e -> ((IntegerAttr) e).value()).toArray();
            default -> throw new BytecodeException("expected integers for " + name + " in " + d);
        };
    }

    private static long[] bounds(String raw) {
        String[] parts = raw.split(",");
        long[] out = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            String p = parts[i].trim();
            out[i] = p.equals("?") ? Long.MIN_VALUE : Long.parseLong(p);
        }
        return out;
    }

    static String type(AttrTypeWriter w, Type t) {
        switch (t) {
            case IntegerType it -> w.varInt(integerCode(it));
            case ComplexType c -> {
                w.varInt(1);
                w.type(c.elementType(), V);
            }
            case FloatType f -> w.varInt(floatCode(f.format()));
            case FunctionType f -> {
                w.varInt(8);
                w.list(f.inputs(), x -> w.type(x, V));
                w.list(f.results(), x -> w.type(x, V));
            }
            case IndexType ignored -> w.varInt(9);
            case TensorType tt -> {
                if (tt.encoding() != null) {
                    w.varInt(21);
                    w.attribute(tt.encoding(), V);
                } else {
                    w.varInt(20);
                }
                long[] dims = tt.dims();
                for (int i = 0; i < dims.length; i++) {
                    if (dims[i] == TensorType.DYNAMIC) {
                        dims[i] = Long.MIN_VALUE;
                    }
                }
                w.signedVarInts(dims);
                w.type(tt.elementType(), V);
            }
            case TokenType ignored -> w.varInt(22);
            case TupleType tu -> {
                w.varInt(23);
                w.list(tu.types(), x -> w.type(x, V));
            }
            case QuantizedType q -> quantized(w, q);
            case FutureType f -> {
                w.varInt(42);
                w.list(f.types(), x -> w.type(x, V));
            }
            case OpaqueType o when o.text().equals("none") -> w.varInt(33);
            case OpaqueType o when o.text().equals("!shape.witness") -> w.varInt(26);
            case OpaqueType o when o.text().startsWith("tensor<*x") -> { // UnrankedTensorV1Type
                w.varInt(25);
                String element = o.text().substring(9, o.text().length() - 1);
                w.type(com.qxotic.jota.stablehlo.text.MlirParser.parseType(element), V);
            }
            case OpaqueType o when o.text().startsWith("memref<") -> { // RankedBufferV1Type
                String body = o.text().substring(7, o.text().length() - 1);
                java.util.List<Long> dims = new java.util.ArrayList<>();
                int p = 0;
                while (true) {
                    int x = body.indexOf('x', p);
                    if (x < 0) {
                        break;
                    }
                    String d = body.substring(p, x);
                    if (d.equals("?")) {
                        dims.add(Long.MIN_VALUE);
                    } else if (d.chars().allMatch(Character::isDigit) && !d.isEmpty()) {
                        dims.add(Long.parseLong(d));
                    } else {
                        break;
                    }
                    p = x + 1;
                }
                w.varInt(41);
                w.signedVarInts(dims.stream().mapToLong(Long::longValue).toArray());
                w.type(com.qxotic.jota.stablehlo.text.MlirParser.parseType(body.substring(p)), V);
            }
            default -> throw new BytecodeException("no VHLO encoding for type " + t.mlir());
        }
        return V;
    }

    private static int integerCode(IntegerType it) {
        if (it.isBoolean()) {
            return 0;
        }
        boolean unsigned = it.isUnsigned();
        return switch (it.bitWidth()) {
            case 2 -> unsigned ? 32 : 31;
            case 4 -> unsigned ? 15 : 10;
            case 8 -> unsigned ? 16 : 11;
            case 16 -> unsigned ? 17 : 12;
            case 32 -> unsigned ? 18 : 13;
            case 64 -> unsigned ? 19 : 14;
            default ->
                    throw new BytecodeException("no VHLO integer type of width " + it.bitWidth());
        };
    }

    private static int floatCode(FloatFormat f) {
        return switch (f) {
            case BF16 -> 2;
            case F16 -> 3;
            case F32 -> 4;
            case F64 -> 5;
            case F8E4M3FN -> 6;
            case F8E5M2 -> 7;
            case F8E4M3FNUZ -> 27;
            case F8E5M2FNUZ -> 28;
            case F8E4M3B11FNUZ -> 29;
            case TF32 -> 34;
            case F8E4M3 -> 35;
            case F8E3M4 -> 36;
            case F4E2M1FN -> 37;
            case F6E2M3FN -> 38;
            case F6E3M2FN -> 39;
            case F8E8M0FNU -> 40;
            case F8E5M3FNU -> 43;
        };
    }

    private static void quantized(AttrTypeWriter w, QuantizedType q) {
        IntegerType storage = q.storageType();
        boolean signed = !storage.isUnsigned();
        int bits = storage.bitWidth();
        long min = q.storageMin() != null ? q.storageMin() : (signed ? -(1L << (bits - 1)) : 0);
        long max =
                q.storageMax() != null
                        ? q.storageMax()
                        : (signed ? (1L << (bits - 1)) - 1 : (1L << bits) - 1);
        if (!q.isPerAxis()) {
            w.varInt(24);
            w.varInt(signed ? 1 : 0);
            w.type(storage, V);
            w.type(q.expressedType(), V);
            w.apInt(Double.doubleToRawLongBits(q.scales().getFirst()), 64);
            w.signedVarInt(q.zeroPoints().getFirst());
            w.signedVarInt(min);
            w.signedVarInt(max);
        } else {
            w.varInt(30);
            w.varInt(signed ? 1 : 0);
            w.type(storage, V);
            w.type(q.expressedType(), V);
            w.varInt(q.quantizationDimension());
            w.signedVarInt(min);
            w.signedVarInt(max);
            w.list(q.scales(), s -> w.apInt(Double.doubleToRawLongBits(s), 64));
            w.list(q.zeroPoints(), w::signedVarInt);
        }
    }
}
