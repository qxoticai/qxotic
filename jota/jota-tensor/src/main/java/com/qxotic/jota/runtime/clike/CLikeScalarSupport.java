package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.DataType;

public final class CLikeScalarSupport {
    private CLikeScalarSupport() {}

    @FunctionalInterface
    public interface CastFn {
        String cast(DataType source, DataType target, String expr);
    }

    @FunctionalInterface
    public interface ToFloatFn {
        String toFloat(DataType source, String expr);
    }

    public static String ternaryExpr(
            String conditionExpr, DataType conditionType, String trueExpr, String falseExpr) {
        String cond = conditionType == DataType.BOOL ? conditionExpr : conditionExpr + " != 0";
        return "(" + cond + " ? " + trueExpr + " : " + falseExpr + ")";
    }

    public static String comparisonExpr(
            String op,
            String leftExpr,
            DataType leftType,
            String rightExpr,
            DataType rightType,
            DataType resultType,
            CastFn castFn,
            ToFloatFn toFloatFn) {
        String left = leftExpr;
        String right = rightExpr;
        DataType lhs = leftType;
        DataType rhs = rightType;

        if (lhs == DataType.FP16 || lhs == DataType.BF16) {
            left = toFloatFn.toFloat(lhs, left);
            lhs = DataType.FP32;
        }
        if (rhs == DataType.FP16 || rhs == DataType.BF16) {
            right = toFloatFn.toFloat(rhs, right);
            rhs = DataType.FP32;
        }

        if (lhs == DataType.BOOL && rhs != DataType.BOOL) {
            left = castFn.cast(lhs, rhs, left);
        } else if (rhs == DataType.BOOL && lhs != DataType.BOOL) {
            right = castFn.cast(rhs, lhs, right);
        }

        String expr = "(" + left + " " + op + " " + right + ")";
        return resultType == DataType.BOOL ? expr : "(" + expr + " ? 1 : 0)";
    }

    public static String boolStoreValue(String valueExpr, DataType valueType) {
        return valueType == DataType.BOOL
                ? "(" + valueExpr + " ? 1 : 0)"
                : "(" + valueExpr + " != 0 ? 1 : 0)";
    }
}
