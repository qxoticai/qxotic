package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
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
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;
import java.util.ArrayList;
import java.util.List;

/**
 * Custom bytecode encodings of the {@code builtin} dialect ({@code
 * mlir/include/mlir/IR/BuiltinDialectBytecode.td}): a leading varint selects the entry by its
 * position in the {@code BuiltinDialectAttributes} / {@code BuiltinDialectTypes} lists.
 */
final class BuiltinDecoder {

    private BuiltinDecoder() {}

    static Attribute attribute(DialectReader r) {
        long kind = r.varInt();
        return switch ((int) kind) {
            case 0 -> new ArrayAttr(r.list(r::attribute)); // ArrayAttr
            case 1 -> { // DictionaryAttr: list of (StringAttr name, Attribute value)
                yield new DictionaryAttr(
                        r.list(() -> MlirBytecodeReader.named(r.attribute(), r.attribute())));
            }
            case 2 -> new StringAttr(r.string()); // StringAttr
            case 3 -> { // StringAttrWithType
                String s = r.string();
                r.type();
                yield new StringAttr(s);
            }
            case 4 -> SymbolRefAttr.of(stringValue(r.attribute())); // FlatSymbolRefAttr
            case 5 -> { // SymbolRefAttr: root + nested flat refs
                List<String> path = new ArrayList<>();
                path.add(stringValue(r.attribute()));
                for (Attribute nested : r.list(r::attribute)) {
                    path.add(((SymbolRefAttr) nested).root());
                }
                yield new SymbolRefAttr(path);
            }
            case 6 -> new TypeAttr(r.type()); // TypeAttr
            case 7 -> UnitAttr.INSTANCE; // UnitAttr
            case 8 -> { // IntegerAttr: type + APInt of the type's width
                Type t = r.type();
                long v = r.apInt(intWidth(t, r));
                // MLIR's BoolAttr is the i1 IntegerAttr (printed true / false)
                yield t.equals(IntegerType.I1) ? BoolAttr.of(v != 0) : new IntegerAttr(v, t);
            }
            case 9 -> { // FloatAttr: type + APFloat
                Type t = r.type();
                if (!(t instanceof FloatType ft)) {
                    throw r.error("float attribute with non-float type " + t);
                }
                yield new FloatAttr(r.apInt(ft.format().bitWidth()), ft);
            }
            case 10 -> skip(r, 2, 0, 0); // CallSiteLoc: callee, caller
            case 11 -> { // FileLineColLoc: filename, line, column
                r.attribute();
                r.varInt();
                r.varInt();
                yield MlirBytecodeReader.LOCATION;
            }
            case 12 -> { // FusedLoc: locations
                r.list(r::attribute);
                yield MlirBytecodeReader.LOCATION;
            }
            case 13 -> { // FusedLocWithMetadata
                r.list(r::attribute);
                r.attribute();
                yield MlirBytecodeReader.LOCATION;
            }
            case 14 -> skip(r, 2, 0, 0); // NameLoc: name, child
            case 15 -> MlirBytecodeReader.LOCATION; // UnknownLoc
            case 16 -> { // DenseResourceElementsAttr: type + resource handle
                Type t = r.type();
                long handle = r.varInt();
                yield new OpaqueAttr("dense_resource<__resource_" + handle + "> : " + t.mlir());
            }
            case 17 -> denseArray(r); // DenseArrayAttr
            case 18 -> denseElements(r, r.type(), r.blob()); // DenseTypedElementsAttr
            case 19 -> { // DenseStringElementsAttr
                Type t = r.type();
                long splat = r.varInt();
                long n = splat != 0 ? 1 : ((TensorType) t).numElements();
                for (long i = 0; i < n; i++) {
                    r.string();
                }
                yield new OpaqueAttr("dense<\"...\"> : " + t.mlir());
            }
            case 21 -> r.attribute(); // DistinctAttr: the referenced attribute
            case 22 -> { // FileLineColRange: filename + list of line/col varints
                r.attribute();
                r.list(r::varInt);
                yield MlirBytecodeReader.LOCATION;
            }
            default -> throw r.error("unsupported builtin attribute encoding " + kind);
        };
    }

    private static Attribute skip(DialectReader r, int attrs, int types, int varInts) {
        for (int i = 0; i < attrs; i++) {
            r.attribute();
        }
        for (int i = 0; i < types; i++) {
            r.type();
        }
        for (int i = 0; i < varInts; i++) {
            r.varInt();
        }
        return MlirBytecodeReader.LOCATION;
    }

    private static String stringValue(Attribute a) {
        if (a instanceof StringAttr s) {
            return s.value();
        }
        throw new BytecodeException("expected a string attribute, got " + a);
    }

    static int intWidth(Type t, DialectReader r) {
        return switch (t) {
            case IntegerType it -> it.bitWidth();
            case IndexType ignored -> 64;
            default -> throw r.error("integer attribute with non-integer type " + t);
        };
    }

    private static Attribute denseArray(DialectReader r) {
        Type et = r.type();
        long size = r.varInt();
        byte[] raw = r.blob();
        int width =
                switch (et) {
                    case IntegerType it -> it.isBoolean() ? 8 : it.bitWidth();
                    case FloatType ft -> ft.format().bitWidth();
                    default -> throw r.error("unsupported dense array element type " + et);
                };
        int bytes = (width + 7) / 8;
        if (raw.length != size * bytes) {
            throw r.error("dense array blob of " + raw.length + " bytes for " + size + " elements");
        }
        long[] values = new long[(int) size];
        for (int i = 0; i < size; i++) {
            long v = 0;
            for (int b = bytes - 1; b >= 0; b--) {
                v = (v << 8) | (raw[i * bytes + b] & 0xFFL);
            }
            if (et instanceof IntegerType it && !it.isUnsigned() && !it.isBoolean() && width < 64) {
                v = (v << (64 - width)) >> (64 - width); // sign-extend
            }
            values[i] = v;
        }
        return new DenseArrayAttr(et, values);
    }

    /**
     * Dense elements from MLIR's raw buffer: elements are byte-rounded little-endian values, a
     * single element means splat; i1 data may be bit-packed (a {@code 0xFF} byte is a true splat).
     */
    static DenseElementsAttr denseElements(DialectReader r, Type type, byte[] raw) {
        if (!(type instanceof TensorType tt)) {
            throw r.error("dense elements need a ranked tensor type, got " + type);
        }
        if (!tt.hasStaticShape()) {
            throw r.error("dense elements need a static shape, got " + tt);
        }
        long n = tt.numElements();
        int elementBytes = tt.elementType().storageBytes();
        if (tt.elementType() instanceof IntegerType it && it.isBoolean() && n > 0) {
            long packed = (n + 7) / 8;
            if (raw.length == 1 && (raw[0] & 0xFF) == 0xFF) {
                return new DenseElementsAttr(tt, new byte[] {1}, true);
            }
            if (raw.length == packed && raw.length != n) {
                byte[] unpacked = new byte[(int) n];
                for (int i = 0; i < n; i++) {
                    unpacked[i] = (byte) ((raw[i / 8] >> (i % 8)) & 1);
                }
                return new DenseElementsAttr(tt, unpacked, false);
            }
        }
        if (raw.length == elementBytes && n != 1) {
            return new DenseElementsAttr(tt, raw, true);
        }
        if (raw.length != n * elementBytes) {
            throw r.error("invalid dense buffer size: " + raw.length + " bytes for " + tt.mlir());
        }
        return new DenseElementsAttr(tt, raw, false);
    }

    static Type type(DialectReader r) {
        long kind = r.varInt();
        return switch ((int) kind) {
            case 0 -> { // IntegerType: (width << 2) | signedness
                long v = r.varInt();
                int width = (int) (v >>> 2);
                IntegerType.Signedness s =
                        switch ((int) (v & 3)) {
                            case 1 -> IntegerType.Signedness.SIGNED;
                            case 2 -> IntegerType.Signedness.UNSIGNED;
                            default -> IntegerType.Signedness.SIGNLESS;
                        };
                yield new IntegerType(width, s);
            }
            case 1 -> IndexType.INSTANCE;
            case 2 -> new FunctionType(r.list(r::type), r.list(r::type));
            case 3 -> FloatType.BF16;
            case 4 -> FloatType.F16;
            case 5 -> FloatType.F32;
            case 6 -> FloatType.F64;
            case 7 -> new OpaqueType("f80");
            case 8 -> new OpaqueType("f128");
            case 9 -> {
                Type et = r.type();
                if (!(et instanceof FloatType ft)) {
                    throw r.error("complex of non-float type " + et);
                }
                yield new ComplexType(ft);
            }
            case 12 -> new OpaqueType("none");
            case 13 -> rankedTensor(r, null);
            case 14 -> rankedTensor(r, r.attribute());
            case 15 -> new TupleType(r.list(r::type));
            case 18 -> new OpaqueType("tensor<*x" + r.type().mlir() + ">");
            case 19 -> {
                long[] shape = r.signedVarInts();
                Type et = r.type();
                StringBuilder sb = new StringBuilder("vector<");
                for (long d : shape) {
                    sb.append(d).append('x');
                }
                yield new OpaqueType(sb.append(et.mlir()).append('>').toString());
            }
            case 21 -> new FloatType(FloatFormat.TF32);
            case 22 -> new FloatType(FloatFormat.F8E5M2);
            case 23 -> new FloatType(FloatFormat.F8E4M3);
            case 24 -> new FloatType(FloatFormat.F8E4M3FN);
            case 25 -> new FloatType(FloatFormat.F8E5M2FNUZ);
            case 26 -> new FloatType(FloatFormat.F8E4M3FNUZ);
            case 27 -> new FloatType(FloatFormat.F8E4M3B11FNUZ);
            case 28 -> new FloatType(FloatFormat.F8E3M4);
            case 29 -> new FloatType(FloatFormat.F4E2M1FN);
            case 30 -> new FloatType(FloatFormat.F6E2M3FN);
            case 31 -> new FloatType(FloatFormat.F6E3M2FN);
            case 32 -> new FloatType(FloatFormat.F8E8M0FNU);
            default -> throw r.error("unsupported builtin type encoding " + kind);
        };
    }

    private static TensorType rankedTensor(DialectReader r, Attribute encoding) {
        long[] shape = r.signedVarInts();
        Type et = r.type();
        if (!(et instanceof ElementType elementType)) {
            throw r.error("unsupported tensor element type " + et);
        }
        List<Long> dims = new ArrayList<>(shape.length);
        for (long d : shape) {
            dims.add(d == Long.MIN_VALUE ? TensorType.DYNAMIC : d);
        }
        return new TensorType(dims, elementType, encoding);
    }
}
