package com.qxotic.jota.stablehlo.tir;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.stablehlo.type.ElementType;
import com.qxotic.jota.stablehlo.type.FloatFormat;
import com.qxotic.jota.stablehlo.type.FloatType;
import com.qxotic.jota.stablehlo.type.IntegerType;
import com.qxotic.jota.stablehlo.type.TensorType;
import com.qxotic.jota.stablehlo.type.Type;

/**
 * StableHLO element types ⇄ TIR {@link DataType}: {@code i1}→BOOL, signless/signed {@code
 * i8..i64}→I8..I64, {@code f16, bf16, f32, f64}. Unsigned, sub-byte, complex, fp8 and quantized
 * types have no TIR counterpart.
 */
public final class TirTypes {

    private TirTypes() {}

    public static DataType dataType(ElementType type, String op) {
        switch (type) {
            case IntegerType it when !it.isUnsigned() -> {
                switch (it.bitWidth()) {
                    case 1:
                        return DataType.BOOL;
                    case 8:
                        return DataType.I8;
                    case 16:
                        return DataType.I16;
                    case 32:
                        return DataType.I32;
                    case 64:
                        return DataType.I64;
                    default:
                        break;
                }
            }
            case FloatType ft when ft.format() == FloatFormat.F16 -> {
                return DataType.FP16;
            }
            case FloatType ft when ft.format() == FloatFormat.BF16 -> {
                return DataType.BF16;
            }
            case FloatType ft when ft.format() == FloatFormat.F32 -> {
                return DataType.FP32;
            }
            case FloatType ft when ft.format() == FloatFormat.F64 -> {
                return DataType.FP64;
            }
            default -> {}
        }
        throw new UnsupportedStableHloException(
                op, "element type " + type.mlir() + " has no TIR type");
    }

    public static TensorType tensorType(Type type, String op) {
        if (type instanceof TensorType tt) {
            if (!tt.hasStaticShape()) {
                throw new UnsupportedStableHloException(op, "dynamic shape " + tt.mlir());
            }
            return tt;
        }
        throw new UnsupportedStableHloException(op, "non-tensor type " + type.mlir());
    }

    public static DataType dataType(Type type, String op) {
        return dataType(tensorType(type, op).elementType(), op);
    }

    public static Shape shape(Type type, String op) {
        return Shape.flat(tensorType(type, op).dims());
    }

    /** Bit width of an integer data type (1 for BOOL). */
    static int bitWidth(DataType dt) {
        return dt == DataType.BOOL ? 1 : (int) dt.byteSize() * 8;
    }
}
