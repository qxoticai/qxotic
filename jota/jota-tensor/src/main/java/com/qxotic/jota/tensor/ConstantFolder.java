package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.ir.tir.BinaryOperator;
import com.qxotic.jota.ir.tir.IRConstantFolder;
import com.qxotic.jota.ir.tir.ScalarConstant;
import com.qxotic.jota.ir.tir.UnaryOperator;
import com.qxotic.jota.runtime.BinaryOp;
import com.qxotic.jota.runtime.UnaryOp;
import java.util.Map;
import java.util.Optional;

final class ConstantFolder {

    private ConstantFolder() {}

    // region Op mapping tables

    private static final Map<BinaryOp, BinaryOperator> BINARY_OP_MAP =
            Map.ofEntries(
                    Map.entry(BinaryOp.ADD, BinaryOperator.ADD),
                    Map.entry(BinaryOp.SUBTRACT, BinaryOperator.SUBTRACT),
                    Map.entry(BinaryOp.MULTIPLY, BinaryOperator.MULTIPLY),
                    Map.entry(BinaryOp.DIVIDE, BinaryOperator.DIVIDE),
                    Map.entry(BinaryOp.MIN, BinaryOperator.MIN),
                    Map.entry(BinaryOp.MAX, BinaryOperator.MAX),
                    Map.entry(BinaryOp.POW, BinaryOperator.POW),
                    Map.entry(BinaryOp.LOGICAL_AND, BinaryOperator.LOGICAL_AND),
                    Map.entry(BinaryOp.LOGICAL_OR, BinaryOperator.LOGICAL_OR),
                    Map.entry(BinaryOp.LOGICAL_XOR, BinaryOperator.LOGICAL_XOR),
                    Map.entry(BinaryOp.BITWISE_AND, BinaryOperator.BITWISE_AND),
                    Map.entry(BinaryOp.BITWISE_OR, BinaryOperator.BITWISE_OR),
                    Map.entry(BinaryOp.BITWISE_XOR, BinaryOperator.BITWISE_XOR),
                    Map.entry(BinaryOp.LEFT_SHIFT, BinaryOperator.SHIFT_LEFT),
                    Map.entry(BinaryOp.RIGHT_SHIFT, BinaryOperator.SHIFT_RIGHT),
                    Map.entry(BinaryOp.RIGHT_SHIFT_UNSIGNED, BinaryOperator.SHIFT_RIGHT_UNSIGNED),
                    Map.entry(BinaryOp.EQUAL, BinaryOperator.EQUAL),
                    Map.entry(BinaryOp.LESS_THAN, BinaryOperator.LESS_THAN));

    private static final Map<UnaryOp, UnaryOperator> UNARY_OP_MAP =
            Map.ofEntries(
                    Map.entry(UnaryOp.NEGATE, UnaryOperator.NEGATE),
                    Map.entry(UnaryOp.ABS, UnaryOperator.ABS),
                    Map.entry(UnaryOp.EXP, UnaryOperator.EXP),
                    Map.entry(UnaryOp.LOG, UnaryOperator.LOG),
                    Map.entry(UnaryOp.SQRT, UnaryOperator.SQRT),
                    Map.entry(UnaryOp.SIN, UnaryOperator.SIN),
                    Map.entry(UnaryOp.COS, UnaryOperator.COS),
                    Map.entry(UnaryOp.TAN, UnaryOperator.TAN),
                    Map.entry(UnaryOp.TANH, UnaryOperator.TANH),
                    Map.entry(UnaryOp.RECIPROCAL, UnaryOperator.RECIPROCAL),
                    Map.entry(UnaryOp.LOGICAL_NOT, UnaryOperator.LOGICAL_NOT),
                    Map.entry(UnaryOp.BITWISE_NOT, UnaryOperator.BITWISE_NOT));

    // endregion Op mapping tables

    static Optional<ConstantComputation> asConstant(Tensor tensor) {
        return InternalTensorAccess.computation(tensor)
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast);
    }

    static Optional<ScalarConstant> asScalarConstant(Tensor tensor) {
        if (tensor instanceof IRTensorImpl ir && ir.node() instanceof ScalarConstant sc) {
            return Optional.of(sc);
        }
        return Optional.empty();
    }

    private static Tensor wrapScalarConstant(ScalarConstant sc, Device device) {
        return new IRTensorImpl(sc, device);
    }

    private static Tensor broadcastedOf(Number value, DataType type, Shape shape) {
        if (shape.isScalar()) {
            if (type.isIntegral() || type == DataType.BOOL) {
                return Tensor.scalar(value.longValue(), type);
            }
            return Tensor.scalar(value.doubleValue(), type);
        }
        return Tensor.full(value, type, shape);
    }

    // region Binary folding

    static Optional<Tensor> tryFoldBinaryOp(Tensor left, Tensor right, BinaryOp op) {
        Optional<ConstantComputation> leftConst = asConstant(left);
        Optional<ConstantComputation> rightConst = asConstant(right);
        if (leftConst.isPresent() && rightConst.isPresent()) {
            ConstantComputation lc = leftConst.get();
            ConstantComputation rc = rightConst.get();
            DataType resultType =
                    TensorTypeSemantics.promoteForArithmetic(
                            lc.dataType(), rc.dataType(), op.name());
            Number result = evalBinary(lc.value(), rc.value(), resultType, op);
            Shape resultShape = lc.shape().size() >= rc.shape().size() ? lc.shape() : rc.shape();
            return Optional.of(broadcastedOf(result, resultType, resultShape));
        }

        Optional<ScalarConstant> leftSc = asScalarConstant(left);
        Optional<ScalarConstant> rightSc = asScalarConstant(right);
        if (leftSc.isPresent() && rightSc.isPresent()) {
            BinaryOperator irOp = BINARY_OP_MAP.get(op);
            if (irOp != null) {
                ScalarConstant folded =
                        IRConstantFolder.foldBinary(irOp, leftSc.get(), rightSc.get());
                if (folded != null) {
                    return Optional.of(wrapScalarConstant(folded, left.device()));
                }
            }
        }

        return Optional.empty();
    }

    static Number evalBinary(Number a, Number b, DataType type, BinaryOp op) {
        if (type == DataType.I8) return (byte) evalIntegral(a.byteValue(), b.byteValue(), 8, op);
        if (type == DataType.I16)
            return (short) evalIntegral(a.shortValue(), b.shortValue(), 16, op);
        if (type == DataType.I32) return (int) evalIntegral(a.intValue(), b.intValue(), 32, op);
        if (type == DataType.I64) return evalIntegral(a.longValue(), b.longValue(), 64, op);
        if (type == DataType.FP16 || type == DataType.BF16 || type == DataType.FP32)
            return (float) evalFPBinary(a.floatValue(), b.floatValue(), op);
        if (type == DataType.FP64) return evalFPBinary(a.doubleValue(), b.doubleValue(), op);
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private static long evalIntegral(long a, long b, int bitWidth, BinaryOp op) {
        return switch (op) {
            case ADD -> a + b;
            case SUBTRACT -> a - b;
            case MULTIPLY -> a * b;
            case DIVIDE -> a / b;
            case MIN -> Math.min(a, b);
            case MAX -> Math.max(a, b);
            case BITWISE_AND -> a & b;
            case BITWISE_OR -> a | b;
            case BITWISE_XOR -> a ^ b;
            case LEFT_SHIFT -> a << (b & (bitWidth - 1));
            case RIGHT_SHIFT -> a >> (b & (bitWidth - 1));
            case RIGHT_SHIFT_UNSIGNED -> {
                int s = (int) (b & (bitWidth - 1));
                yield switch (bitWidth) {
                    case 8 -> (a & 0xFFL) >>> s;
                    case 16 -> (a & 0xFFFFL) >>> s;
                    case 32 -> (a & 0xFFFFFFFFL) >>> s;
                    default -> a >>> s;
                };
            }
            default -> throw new UnsupportedOperationException("Cannot fold: " + op);
        };
    }

    private static double evalFPBinary(double a, double b, BinaryOp op) {
        return switch (op) {
            case ADD -> a + b;
            case SUBTRACT -> a - b;
            case MULTIPLY -> a * b;
            case DIVIDE -> a / b;
            case MIN -> Math.min(a, b);
            case MAX -> Math.max(a, b);
            default -> throw new UnsupportedOperationException("Cannot fold: " + op);
        };
    }

    // endregion Binary folding

    // region Unary folding

    static Optional<Tensor> tryFoldUnaryOp(Tensor tensor, UnaryOp op) {
        Optional<ConstantComputation> constant = asConstant(tensor);
        if (constant.isPresent()) {
            ConstantComputation c = constant.get();
            try {
                Number result = evalUnary(c.value(), c.dataType(), op);
                return Optional.of(broadcastedOf(result, c.dataType(), c.shape()));
            } catch (UnsupportedOperationException e) {
            }
        }

        Optional<ScalarConstant> sc = asScalarConstant(tensor);
        if (sc.isPresent()) {
            UnaryOperator irOp = UNARY_OP_MAP.get(op);
            if (irOp != null) {
                ScalarConstant folded = IRConstantFolder.foldUnary(irOp, sc.get());
                if (folded != null) {
                    return Optional.of(wrapScalarConstant(folded, tensor.device()));
                }
            }
        }

        return Optional.empty();
    }

    static Number evalUnary(Number a, DataType type, UnaryOp op) {
        if (type == DataType.BOOL) return (long) evalBool(a.longValue() != 0, op);
        if (type.isIntegral()) return narrowIntegral(evalIntegralUnary(a.longValue(), op), type);
        if (type == DataType.FP64) return evalFPUnary(a.doubleValue(), op);
        // FP16, BF16, FP32: compute in double, return as float
        if (type == DataType.FP16 || type == DataType.BF16 || type == DataType.FP32)
            return (float) evalFPUnary(a.floatValue(), op);
        throw new UnsupportedOperationException("Unsupported type: " + type);
    }

    private static long evalIntegralUnary(long a, UnaryOp op) {
        return switch (op) {
            case NEGATE -> -a;
            case ABS -> Math.abs(a);
            case BITWISE_NOT -> ~a;
            default -> throw new UnsupportedOperationException("Unsupported unary op: " + op);
        };
    }

    private static Number narrowIntegral(long value, DataType type) {
        if (type == DataType.I8) return (byte) value;
        if (type == DataType.I16) return (short) value;
        if (type == DataType.I32) return (int) value;
        return value;
    }

    private static double evalFPUnary(double a, UnaryOp op) {
        return switch (op) {
            case NEGATE -> -a;
            case ABS -> Math.abs(a);
            case EXP -> Math.exp(a);
            case LOG -> Math.log(a);
            case SQRT -> Math.sqrt(a);
            case SIN -> Math.sin(a);
            case COS -> Math.cos(a);
            case TANH -> Math.tanh(a);
            case RECIPROCAL -> 1.0 / a;
            default -> throw new UnsupportedOperationException("Unsupported unary op: " + op);
        };
    }

    private static byte evalBool(boolean a, UnaryOp op) {
        if (op == UnaryOp.LOGICAL_NOT) return (byte) (a ? 0 : 1);
        throw new UnsupportedOperationException("Unsupported unary op for BOOL: " + op);
    }

    // endregion Unary folding

    // region Compare folding

    static Optional<Tensor> tryFoldCompareOp(Tensor left, Tensor right, BinaryOp op) {
        Optional<ConstantComputation> leftConst = asConstant(left);
        Optional<ConstantComputation> rightConst = asConstant(right);
        if (leftConst.isEmpty() || rightConst.isEmpty()) {
            return Optional.empty();
        }
        ConstantComputation lc = leftConst.get();
        ConstantComputation rc = rightConst.get();
        DataType commonType =
                TensorTypeSemantics.promoteForComparison(lc.dataType(), rc.dataType(), op.name());
        boolean result = evalCompare(lc.value(), rc.value(), commonType, op);
        return Optional.of(Tensor.scalar(result ? 1L : 0L, DataType.BOOL));
    }

    static boolean evalCompare(Number a, Number b, DataType type, BinaryOp op) {
        if (type.isIntegral() || type == DataType.BOOL) {
            return evalCompareIntegral(a.longValue(), b.longValue(), op);
        }
        return evalCompareFP(a.doubleValue(), b.doubleValue(), op);
    }

    private static boolean evalCompareIntegral(long a, long b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    private static boolean evalCompareFP(double a, double b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    // endregion Compare folding

    // region Cast folding

    static Optional<Tensor> tryFoldCast(Tensor tensor, DataType targetType) {
        Optional<ConstantComputation> constant = asConstant(tensor);
        if (constant.isPresent()) {
            ConstantComputation c = constant.get();
            Number result = cast(c.value(), targetType);
            return Optional.of(broadcastedOf(result, targetType, c.shape()));
        }

        Optional<ScalarConstant> sc = asScalarConstant(tensor);
        if (sc.isPresent()) {
            ScalarConstant folded = IRConstantFolder.foldCast(sc.get(), targetType);
            if (folded != null) {
                return Optional.of(wrapScalarConstant(folded, tensor.device()));
            }
        }

        return Optional.empty();
    }

    static Number cast(Number value, DataType targetType) {
        if (targetType == DataType.I8) return value.byteValue();
        if (targetType == DataType.I16) return value.shortValue();
        if (targetType == DataType.I32) return value.intValue();
        if (targetType == DataType.I64) return value.longValue();
        if (targetType == DataType.FP16
                || targetType == DataType.BF16
                || targetType == DataType.FP32) return value.floatValue();
        if (targetType == DataType.FP64) return value.doubleValue();
        if (targetType == DataType.BOOL) return value.longValue() != 0 ? 1L : 0L;
        throw new UnsupportedOperationException("Unsupported target type: " + targetType);
    }

    // endregion Cast folding
}
