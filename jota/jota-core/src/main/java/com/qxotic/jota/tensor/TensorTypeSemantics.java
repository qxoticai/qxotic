package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.TypeRules;

final class TensorTypeSemantics {

    private TensorTypeSemantics() {}

    static DataType promoteForArithmetic(DataType left, DataType right, String opName) {
        try {
            return TypeRules.promote(left, right);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    opName + " does not support dtypes " + left + " and " + right, e);
        }
    }

    static DataType promoteForComparison(DataType left, DataType right, String opName) {
        try {
            return TypeRules.promoteForComparison(left, right);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                    opName + " does not support dtypes " + left + " and " + right, e);
        }
    }

    static void requireSameIntegralType(DataType left, DataType right, String opName) {
        if (left != right) {
            throw new IllegalArgumentException(
                    opName
                            + " requires operands with the same data type, got "
                            + left
                            + " and "
                            + right);
        }
        requireIntegral(left, opName);
    }

    static void requireShiftOperandTypes(DataType valueType, DataType shiftType, String opName) {
        requireIntegral(valueType, opName);
        if (shiftType.isIntegral() && shiftType != DataType.BOOL) {
            return;
        }
        throw new IllegalArgumentException(
                opName + " requires integral shift counts (non-BOOL), got " + shiftType);
    }

    static void requireBooleanPair(DataType left, DataType right, String opName) {
        if (left != DataType.BOOL || right != DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires BOOL tensors, got " + left + " and " + right);
        }
    }

    static void requireIntegral(DataType dataType, String opName) {
        if (!dataType.isIntegral() || dataType == DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires integral data type, got " + dataType);
        }
    }

    static void requireFloatingPoint(DataType dataType, String opName) {
        if (!dataType.isFloatingPoint()) {
            throw new IllegalArgumentException(
                    opName + " requires floating-point tensor, got " + dataType);
        }
    }

    static void requireNumericNonBool(DataType dataType, String opName) {
        if (dataType == DataType.BOOL || (!dataType.isIntegral() && !dataType.isFloatingPoint())) {
            throw new IllegalArgumentException(
                    opName + " requires numeric non-BOOL tensor, got " + dataType);
        }
    }

    static void requireBool(DataType dataType, String opName) {
        if (dataType != DataType.BOOL) {
            throw new IllegalArgumentException(
                    opName + " requires BOOL data type, got " + dataType);
        }
    }

    static DataType resolveReductionAccumulator(
            DataType inputType, DataType accumulatorType, String opName) {
        if (accumulatorType == null) {
            return inputType;
        }
        if (accumulatorType == DataType.BOOL
                || (!accumulatorType.isIntegral() && !accumulatorType.isFloatingPoint())) {
            throw new IllegalArgumentException(
                    opName + " accumulator must be numeric (non-BOOL), got " + accumulatorType);
        }
        if (inputType == DataType.BOOL) {
            return accumulatorType;
        }
        DataType promoted =
                promoteForArithmetic(inputType, accumulatorType, opName + " accumulator");
        if (promoted != accumulatorType) {
            throw new IllegalArgumentException(
                    opName
                            + " accumulator "
                            + accumulatorType
                            + " cannot safely represent input "
                            + inputType);
        }
        return accumulatorType;
    }
}
