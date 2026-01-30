package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;

/**
 * Constant folding for IR-T scalar constants.
 *
 * <p>Folds operations on {@link ScalarConstant} nodes at IR construction time, avoiding unnecessary
 * compute nodes when the result can be determined statically.
 */
public final class IRConstantFolder {

    private IRConstantFolder() {}

    /** Attempts to fold a unary operation on a scalar constant. Returns null if not foldable. */
    public static ScalarConstant foldUnary(UnaryOperator op, ScalarConstant input) {
        DataType dataType = input.dataType();
        Shape shape = input.layout().shape();

        if (dataType == DataType.FP32) {
            float value = Float.intBitsToFloat((int) input.rawBits());
            Float result = applyUnaryFloat(op, value);
            if (result != null) {
                return ScalarConstant.broadcast(Float.floatToIntBits(result), dataType, shape);
            }
        } else if (dataType == DataType.FP64) {
            double value = Double.longBitsToDouble(input.rawBits());
            Double result = applyUnaryDouble(op, value);
            if (result != null) {
                return ScalarConstant.broadcast(Double.doubleToLongBits(result), dataType, shape);
            }
        } else if (dataType == DataType.I32) {
            int value = (int) input.rawBits();
            Integer result = applyUnaryInt(op, value);
            if (result != null) {
                return ScalarConstant.broadcast(result, dataType, shape);
            }
        } else if (dataType == DataType.I64) {
            long value = input.rawBits();
            Long result = applyUnaryLong(op, value);
            if (result != null) {
                return ScalarConstant.broadcast(result, dataType, shape);
            }
        }

        return null;
    }

    /**
     * Attempts to fold a binary operation on two scalar constants. Returns null if not foldable.
     */
    public static ScalarConstant foldBinary(
            BinaryOperator op, ScalarConstant left, ScalarConstant right) {
        DataType dataType = left.dataType();
        Shape shape = left.layout().shape();

        // Both must have same dataType (caller should have promoted)
        if (left.dataType() != right.dataType()) {
            return null;
        }

        if (dataType == DataType.FP32) {
            float lv = Float.intBitsToFloat((int) left.rawBits());
            float rv = Float.intBitsToFloat((int) right.rawBits());
            Float result = applyBinaryFloat(op, lv, rv);
            if (result != null) {
                return ScalarConstant.broadcast(Float.floatToIntBits(result), dataType, shape);
            }
        } else if (dataType == DataType.FP64) {
            double lv = Double.longBitsToDouble(left.rawBits());
            double rv = Double.longBitsToDouble(right.rawBits());
            Double result = applyBinaryDouble(op, lv, rv);
            if (result != null) {
                return ScalarConstant.broadcast(Double.doubleToLongBits(result), dataType, shape);
            }
        } else if (dataType == DataType.I32) {
            int lv = (int) left.rawBits();
            int rv = (int) right.rawBits();
            Integer result = applyBinaryInt(op, lv, rv);
            if (result != null) {
                return ScalarConstant.broadcast(result, dataType, shape);
            }
        } else if (dataType == DataType.I64) {
            long lv = left.rawBits();
            long rv = right.rawBits();
            Long result = applyBinaryLong(op, lv, rv);
            if (result != null) {
                return ScalarConstant.broadcast(result, dataType, shape);
            }
        }

        return null;
    }

    /** Attempts to fold a cast operation on a scalar constant. Returns null if not foldable. */
    public static ScalarConstant foldCast(ScalarConstant input, DataType targetType) {
        if (input.dataType() == targetType) {
            return input;
        }

        DataType srcType = input.dataType();
        Shape shape = input.layout().shape();

        // Extract source value as double (for floats) or long (for ints)
        double floatValue;
        long intValue;

        if (srcType == DataType.FP32) {
            floatValue = Float.intBitsToFloat((int) input.rawBits());
            intValue = (long) floatValue;
        } else if (srcType == DataType.FP64) {
            floatValue = Double.longBitsToDouble(input.rawBits());
            intValue = (long) floatValue;
        } else if (srcType == DataType.I32) {
            intValue = (int) input.rawBits();
            floatValue = intValue;
        } else if (srcType == DataType.I64) {
            intValue = input.rawBits();
            floatValue = intValue;
        } else if (srcType == DataType.I8) {
            intValue = (byte) input.rawBits();
            floatValue = intValue;
        } else if (srcType == DataType.I16) {
            intValue = (short) input.rawBits();
            floatValue = intValue;
        } else {
            return null; // Unsupported source type
        }

        // Convert to target type
        long rawBits;
        if (targetType == DataType.FP32) {
            rawBits = Float.floatToIntBits((float) floatValue);
        } else if (targetType == DataType.FP64) {
            rawBits = Double.doubleToLongBits(floatValue);
        } else if (targetType == DataType.I32) {
            rawBits = (int) intValue;
        } else if (targetType == DataType.I64) {
            rawBits = intValue;
        } else if (targetType == DataType.I8) {
            rawBits = (byte) intValue;
        } else if (targetType == DataType.I16) {
            rawBits = (short) intValue;
        } else {
            return null; // Unsupported target type
        }

        return ScalarConstant.broadcast(rawBits, targetType, shape);
    }

    // ==================== Float Operations ====================

    private static Float applyUnaryFloat(UnaryOperator op, float value) {
        return switch (op) {
            case NEGATE -> -value;
            case ABS -> Math.abs(value);
            case EXP -> (float) Math.exp(value);
            case LOG -> (float) Math.log(value);
            case SQRT -> (float) Math.sqrt(value);
            case SQUARE -> value * value;
            case SIN -> (float) Math.sin(value);
            case COS -> (float) Math.cos(value);
            case TAN -> (float) Math.tan(value);
            case TANH -> (float) Math.tanh(value);
            case RECIPROCAL -> 1.0f / value;
            default -> null;
        };
    }

    private static Float applyBinaryFloat(BinaryOperator op, float left, float right) {
        return switch (op) {
            case ADD -> left + right;
            case SUBTRACT -> left - right;
            case MULTIPLY -> left * right;
            case DIVIDE -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW -> (float) Math.pow(left, right);
            default -> null;
        };
    }

    // ==================== Double Operations ====================

    private static Double applyUnaryDouble(UnaryOperator op, double value) {
        return switch (op) {
            case NEGATE -> -value;
            case ABS -> Math.abs(value);
            case EXP -> Math.exp(value);
            case LOG -> Math.log(value);
            case SQRT -> Math.sqrt(value);
            case SQUARE -> value * value;
            case SIN -> Math.sin(value);
            case COS -> Math.cos(value);
            case TAN -> Math.tan(value);
            case TANH -> Math.tanh(value);
            case RECIPROCAL -> 1.0 / value;
            default -> null;
        };
    }

    private static Double applyBinaryDouble(BinaryOperator op, double left, double right) {
        return switch (op) {
            case ADD -> left + right;
            case SUBTRACT -> left - right;
            case MULTIPLY -> left * right;
            case DIVIDE -> left / right;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case POW -> Math.pow(left, right);
            default -> null;
        };
    }

    // ==================== Int Operations ====================

    private static Integer applyUnaryInt(UnaryOperator op, int value) {
        return switch (op) {
            case NEGATE -> -value;
            case ABS -> Math.abs(value);
            case BITWISE_NOT -> ~value;
            default -> null;
        };
    }

    private static Integer applyBinaryInt(BinaryOperator op, int left, int right) {
        return switch (op) {
            case ADD -> left + right;
            case SUBTRACT -> left - right;
            case MULTIPLY -> left * right;
            case DIVIDE -> right != 0 ? left / right : null;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case BITWISE_AND -> left & right;
            case BITWISE_OR -> left | right;
            case BITWISE_XOR -> left ^ right;
            default -> null;
        };
    }

    // ==================== Long Operations ====================

    private static Long applyUnaryLong(UnaryOperator op, long value) {
        return switch (op) {
            case NEGATE -> -value;
            case ABS -> Math.abs(value);
            case BITWISE_NOT -> ~value;
            default -> null;
        };
    }

    private static Long applyBinaryLong(BinaryOperator op, long left, long right) {
        return switch (op) {
            case ADD -> left + right;
            case SUBTRACT -> left - right;
            case MULTIPLY -> left * right;
            case DIVIDE -> right != 0 ? left / right : null;
            case MIN -> Math.min(left, right);
            case MAX -> Math.max(left, right);
            case BITWISE_AND -> left & right;
            case BITWISE_OR -> left | right;
            case BITWISE_XOR -> left ^ right;
            default -> null;
        };
    }
}
