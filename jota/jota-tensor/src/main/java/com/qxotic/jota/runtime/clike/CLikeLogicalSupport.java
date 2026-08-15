package com.qxotic.jota.runtime.clike;

import com.qxotic.jota.DataType;

public final class CLikeLogicalSupport {
    private CLikeLogicalSupport() {}

    public static String normalizeLogicalOperand(String expr, DataType operandType) {
        return operandType != DataType.BOOL ? "(" + expr + " != 0)" : expr;
    }

    public static String logicalBinary(
            String op,
            String leftExpr,
            DataType leftType,
            String rightExpr,
            DataType rightType,
            DataType resultType) {
        String left = normalizeLogicalOperand(leftExpr, leftType);
        String right = normalizeLogicalOperand(rightExpr, rightType);
        return logicalResult("(" + left + " " + op + " " + right + ")", resultType);
    }

    public static String logicalXor(
            String leftExpr,
            DataType leftType,
            String rightExpr,
            DataType rightType,
            DataType resultType) {
        String left = normalizeLogicalOperand(leftExpr, leftType);
        String right = normalizeLogicalOperand(rightExpr, rightType);
        return logicalResult("(" + left + " ^ " + right + ")", resultType);
    }

    public static String maybeConvertBoolToNumeric(
            String expr, DataType operandType, DataType targetType) {
        if (operandType == DataType.BOOL && targetType != DataType.BOOL) {
            return "(" + expr + " ? 1 : 0)";
        }
        return expr;
    }

    private static String logicalResult(String boolExpr, DataType resultType) {
        return resultType == DataType.BOOL ? boolExpr : "(" + boolExpr + " ? 1 : 0)";
    }
}
