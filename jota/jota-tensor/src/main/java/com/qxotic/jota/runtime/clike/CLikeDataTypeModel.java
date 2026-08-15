package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.DataType;
import java.util.Map;

public final class CLikeDataTypeModel {

    private final Map<DataType, String> typeNames;
    private final Map<DataType, String> unsignedShiftTypes;

    public CLikeDataTypeModel(
            Map<DataType, String> typeNames, Map<DataType, String> unsignedShiftTypes) {
        this.typeNames = Map.copyOf(typeNames);
        this.unsignedShiftTypes = Map.copyOf(unsignedShiftTypes);
    }

    public String renderType(DataType dataType) {
        String renderedType = typeNames.get(dataType);
        if (renderedType == null) {
            throw new UnsupportedOperationException("Unsupported data type: " + dataType);
        }
        return renderedType;
    }

    public String renderLiteral(long bits, DataType type) {
        if (type == DataType.FP32) {
            return CLikeExprSupport.formatFloatLiteral(Float.intBitsToFloat((int) bits));
        }
        if (type == DataType.FP64) {
            return CLikeExprSupport.formatDoubleLiteral(Double.longBitsToDouble(bits));
        }
        if (type == DataType.FP16) {
            String value = CLikeExprSupport.formatFloatLiteral(Float.float16ToFloat((short) bits));
            return "(" + renderType(DataType.FP16) + ")(" + value + ")";
        }
        if (type == DataType.BF16) {
            int bf16Bits = ((int) bits) & 0xFFFF;
            String value =
                    CLikeExprSupport.formatFloatLiteral(Float.intBitsToFloat(bf16Bits << 16));
            return "(" + renderType(DataType.BF16) + ")(" + value + ")";
        }
        if (type == DataType.I8) {
            return "(" + renderType(DataType.I8) + ")" + bits + "LL";
        }
        if (type == DataType.I16) {
            return "(" + renderType(DataType.I16) + ")" + bits + "LL";
        }
        if (type == DataType.I32) {
            return "(" + renderType(DataType.I32) + ")" + bits + "LL";
        }
        if (type == DataType.I64) {
            return "(" + renderType(DataType.I64) + ")" + bits + "LL";
        }
        if (type == DataType.BOOL) {
            return bits != 0 ? "1" : "0";
        }
        throw new UnsupportedOperationException("Unsupported literal dtype: " + type);
    }

    public String renderToFloat32(DataType source, String expr) {
        if (source == DataType.FP32) {
            return expr;
        }
        if (source == DataType.BOOL) {
            return "(" + expr + " != 0 ? 1.0f : 0.0f)";
        }
        return "(float)(" + expr + ")";
    }

    public String renderToFloat64(DataType source, String expr) {
        if (source == DataType.FP64) {
            return expr;
        }
        if (source == DataType.FP16 || source == DataType.BF16) {
            return "(double)(" + renderToFloat32(source, expr) + ")";
        }
        if (source == DataType.BOOL) {
            return "(" + expr + " != 0 ? 1.0 : 0.0)";
        }
        return "(double)(" + expr + ")";
    }

    public String renderUnsignedShiftCarrierType(DataType type) {
        String renderedType = unsignedShiftTypes.get(type);
        if (renderedType == null) {
            throw new UnsupportedOperationException("Unsupported shift type: " + type);
        }
        return renderedType;
    }
}
