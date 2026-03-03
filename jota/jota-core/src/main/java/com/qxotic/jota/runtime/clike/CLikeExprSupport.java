package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.DataType;
import com.qxotic.jota.ir.lir.IndexBinaryOp;

public final class CLikeExprSupport {
    private CLikeExprSupport() {}

    public static String normalizedShift(DataType type, String right) {
        int mask;
        if (type == DataType.I8) {
            mask = 7;
        } else if (type == DataType.I16) {
            mask = 15;
        } else if (type == DataType.I64) {
            mask = 63;
        } else {
            mask = 31;
        }
        return "((int)(" + right + ") & " + mask + ")";
    }

    public static String indexOp(IndexBinaryOp op) {
        return switch (op) {
            case ADD -> "+";
            case SUBTRACT -> "-";
            case MULTIPLY -> "*";
            case DIVIDE -> "/";
            case MODULO -> "%";
            case BITWISE_AND -> "&";
            case BITWISE_XOR -> "^";
            case SHIFT_LEFT -> "<<";
            case SHIFT_RIGHT -> ">>";
            case UNSIGNED_SHIFT_RIGHT -> ">>";
        };
    }

    public static String formatFloatLiteral(float value) {
        if (Float.isNaN(value)) {
            return "NAN";
        }
        if (Float.isInfinite(value)) {
            return value > 0 ? "INFINITY" : "-INFINITY";
        }
        return ensureDecimal(Float.toString(value)) + "f";
    }

    public static String formatDoubleLiteral(double value) {
        if (Double.isNaN(value)) {
            return "NAN";
        }
        if (Double.isInfinite(value)) {
            return value > 0 ? "INFINITY" : "-INFINITY";
        }
        return ensureDecimal(Double.toString(value));
    }

    private static String ensureDecimal(String value) {
        if (value.contains("E") || value.contains("e") || value.contains(".")) {
            return value;
        }
        return value + ".0";
    }
}
