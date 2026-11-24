package com.qxotic.jota.stablehlo.bytecode;

import com.qxotic.jota.stablehlo.attr.ArrayAttr;
import com.qxotic.jota.stablehlo.attr.Attribute;
import com.qxotic.jota.stablehlo.attr.BoolAttr;
import com.qxotic.jota.stablehlo.attr.DenseArrayAttr;
import com.qxotic.jota.stablehlo.attr.DenseElementsAttr;
import com.qxotic.jota.stablehlo.attr.DictionaryAttr;
import com.qxotic.jota.stablehlo.attr.FloatAttr;
import com.qxotic.jota.stablehlo.attr.IntegerAttr;
import com.qxotic.jota.stablehlo.attr.NamedAttribute;
import com.qxotic.jota.stablehlo.attr.OpaqueAttr;
import com.qxotic.jota.stablehlo.attr.StringAttr;
import com.qxotic.jota.stablehlo.attr.SymbolRefAttr;
import com.qxotic.jota.stablehlo.attr.TypeAttr;
import com.qxotic.jota.stablehlo.attr.UnitAttr;
import com.qxotic.jota.stablehlo.type.ComplexType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.FunctionType;
import com.qxotic.jota.stablehlo.type.IndexType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.OpaqueType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.TupleType;
import com.qxotic.jota.stablehlo.type.Type;

/**
 * Builtin dialect encodings (inverse of {@link BuiltinDecoder}); anything without a builtin custom
 * encoding is written in assembly form under its own dialect.
 */
final class BuiltinEncoder {

    static final int DICTIONARY = 1;
    private static final String B = MlirBytecodeWriter.BUILTIN;

    private BuiltinEncoder() {}

    /** Encodes {@code a}; returns the dialect owning the entry. */
    static String attribute(AttrTypeWriter w, Attribute a) {
        switch (a) {
            case OpaqueAttr o when o.text().equals("loc(unknown)") -> w.varInt(15);
            case ArrayAttr arr -> {
                w.varInt(0);
                w.list(arr.elements(), e -> w.attribute(e, B));
            }
            case DictionaryAttr d -> {
                w.varInt(DICTIONARY);
                w.list(
                        d.entries(),
                        (NamedAttribute e) -> {
                            w.attribute(new StringAttr(e.name()), B);
                            w.attribute(e.value(), B);
                        });
            }
            case StringAttr s -> {
                w.varInt(2);
                w.string(s.value());
            }
            case SymbolRefAttr s when s.path().size() == 1 -> {
                w.varInt(4);
                w.attribute(new StringAttr(s.root()), B);
            }
            case SymbolRefAttr s -> {
                w.varInt(5);
                w.attribute(new StringAttr(s.root()), B);
                w.list(
                        s.path().subList(1, s.path().size()),
                        n -> w.attribute(SymbolRefAttr.of(n), B));
            }
            case TypeAttr t -> {
                w.varInt(6);
                w.type(t.type(), B);
            }
            case UnitAttr u -> w.varInt(7);
            case BoolAttr b -> {
                w.varInt(8);
                w.type(IntegerType.I1, B);
                w.apInt(b.value() ? 1 : 0, 1);
            }
            case IntegerAttr i -> {
                w.varInt(8);
                w.type(i.type(), B);
                w.apInt(i.value(), i.type() instanceof IntegerType it ? it.bitWidth() : 64);
            }
            case FloatAttr f -> {
                w.varInt(9);
                w.type(f.type(), B);
                w.apInt(f.bits(), f.type().format().bitWidth());
            }
            case DenseArrayAttr d -> {
                w.varInt(17);
                w.type(d.elementType(), B);
                w.varInt(d.size());
                int width =
                        switch (d.elementType()) {
                            case IntegerType it -> it.isBoolean() ? 8 : it.bitWidth();
                            case FloatType ft -> ft.format().bitWidth();
                            default ->
                                    throw new BytecodeException(
                                            "dense array of " + d.elementType());
                        };
                int bytes = (width + 7) / 8;
                byte[] raw = new byte[d.size() * bytes];
                for (int i = 0; i < d.size(); i++) {
                    for (int k = 0; k < bytes; k++) {
                        raw[i * bytes + k] = (byte) (d.get(i) >>> (8 * k));
                    }
                }
                w.blob(raw);
            }
            case DenseElementsAttr d -> {
                w.varInt(18);
                w.type(d.type(), B);
                w.blob(denseBlob(d));
            }
            default -> {
                w.assembly(a.toString());
                return dialectOfAssembly(a.toString());
            }
        }
        return B;
    }

    /** MLIR's raw dense buffer; i1 data is bit-packed (a splat is {@code 0xFF} / {@code 0x00}). */
    static byte[] denseBlob(DenseElementsAttr d) {
        byte[] raw = d.rawData();
        if (!(d.type().elementType() instanceof IntegerType it && it.isBoolean())) {
            return raw;
        }
        long n = d.numElements();
        if (d.isSplat() && n > 1) {
            return new byte[] {(byte) (raw[0] != 0 ? 0xFF : 0x00)};
        }
        byte[] packed = new byte[(int) ((n + 7) / 8)];
        for (int i = 0; i < n; i++) {
            if (d.getBoolean(i)) {
                packed[i / 8] |= (byte) (1 << (i % 8));
            }
        }
        return packed;
    }

    /** {@code #stablehlo.x<...>} / {@code !stablehlo.token} → {@code stablehlo}. */
    static String dialectOfAssembly(String text) {
        if (text.startsWith("#") || text.startsWith("!")) {
            int end = 1;
            while (end < text.length() && Character.isLetterOrDigit(text.charAt(end))) {
                end++;
            }
            return text.substring(1, end);
        }
        return B;
    }

    static String type(AttrTypeWriter w, Type t) {
        switch (t) {
            case IntegerType it -> {
                w.varInt(0);
                int signedness =
                        switch (it.signedness()) {
                            case SIGNLESS -> 0;
                            case SIGNED -> 1;
                            case UNSIGNED -> 2;
                        };
                w.varInt(((long) it.bitWidth() << 2) | signedness);
            }
            case IndexType ignored -> w.varInt(1);
            case FunctionType f -> {
                w.varInt(2);
                w.list(f.inputs(), x -> w.type(x, B));
                w.list(f.results(), x -> w.type(x, B));
            }
            case FloatType f when f.format() != FloatFormat.F8E5M3FNU ->
                    w.varInt(floatCode(f.format()));
            case ComplexType c -> {
                w.varInt(9);
                w.type(c.elementType(), B);
            }
            case OpaqueType o when o.text().equals("none") -> w.varInt(12);
            case TensorType tt -> {
                if (tt.encoding() != null) {
                    w.varInt(14);
                    w.attribute(tt.encoding(), B);
                } else {
                    w.varInt(13);
                }
                long[] dims = tt.dims();
                for (int i = 0; i < dims.length; i++) {
                    if (dims[i] == TensorType.DYNAMIC) {
                        dims[i] = Long.MIN_VALUE;
                    }
                }
                w.signedVarInts(dims);
                w.type(tt.elementType(), B);
            }
            case TupleType tu -> {
                w.varInt(15);
                w.list(tu.types(), x -> w.type(x, B));
            }
            default -> {
                w.assembly(t.mlir());
                return dialectOfAssembly(t.mlir());
            }
        }
        return B;
    }

    private static int floatCode(FloatFormat f) {
        return switch (f) {
            case BF16 -> 3;
            case F16 -> 4;
            case F32 -> 5;
            case F64 -> 6;
            case TF32 -> 21;
            case F8E5M2 -> 22;
            case F8E4M3 -> 23;
            case F8E4M3FN -> 24;
            case F8E5M2FNUZ -> 25;
            case F8E4M3FNUZ -> 26;
            case F8E4M3B11FNUZ -> 27;
            case F8E3M4 -> 28;
            case F4E2M1FN -> 29;
            case F6E2M3FN -> 30;
            case F6E3M2FN -> 31;
            case F8E8M0FNU -> 32;
            case F8E5M3FNU ->
                    throw new BytecodeException("no builtin bytecode encoding for f8E5M3FNU");
        };
    }
}
