package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.ir.tir.BinaryOperator;
import ai.qxotic.jota.ir.tir.IRConstantFolder;
import ai.qxotic.jota.ir.tir.ScalarConstant;
import ai.qxotic.jota.ir.tir.UnaryOperator;
import java.util.Optional;

/** Package-private utility class for constant folding of tensor operations. */
final class ConstantFolder {

    private ConstantFolder() {}

    static Optional<ConstantComputation> asConstant(Tensor tensor) {
        return tensor.computation()
                .filter(ConstantComputation.class::isInstance)
                .map(ConstantComputation.class::cast);
    }

    /** Returns the ScalarConstant node if tensor is an IRTensor wrapping one. */
    static Optional<ScalarConstant> asScalarConstant(Tensor tensor) {
        if (tensor instanceof IRTensor ir && ir.node() instanceof ScalarConstant sc) {
            return Optional.of(sc);
        }
        return Optional.empty();
    }

    /** Wraps a ScalarConstant in an IRTensor with the given device. */
    private static IRTensor wrapScalarConstant(ScalarConstant sc, Device device) {
        return new IRTensor(sc, device);
    }

    /** Maps Tensor API BinaryOp to IR BinaryOperator. Returns null if not mappable. */
    private static BinaryOperator toIRBinaryOp(BinaryOp op) {
        if (op == BinaryOp.ADD) return BinaryOperator.ADD;
        if (op == BinaryOp.SUBTRACT) return BinaryOperator.SUBTRACT;
        if (op == BinaryOp.MULTIPLY) return BinaryOperator.MULTIPLY;
        if (op == BinaryOp.DIVIDE) return BinaryOperator.DIVIDE;
        if (op == BinaryOp.MIN) return BinaryOperator.MIN;
        if (op == BinaryOp.MAX) return BinaryOperator.MAX;
        if (op == BinaryOp.POW) return BinaryOperator.POW;
        if (op == BinaryOp.LOGICAL_AND) return BinaryOperator.LOGICAL_AND;
        if (op == BinaryOp.LOGICAL_OR) return BinaryOperator.LOGICAL_OR;
        if (op == BinaryOp.LOGICAL_XOR) return BinaryOperator.LOGICAL_XOR;
        if (op == BinaryOp.BITWISE_AND) return BinaryOperator.BITWISE_AND;
        if (op == BinaryOp.BITWISE_OR) return BinaryOperator.BITWISE_OR;
        if (op == BinaryOp.BITWISE_XOR) return BinaryOperator.BITWISE_XOR;
        if (op == BinaryOp.EQUAL) return BinaryOperator.EQUAL;
        if (op == BinaryOp.LESS_THAN) return BinaryOperator.LESS_THAN;
        return null;
    }

    /** Maps Tensor API UnaryOp to IR UnaryOperator. Returns null if not mappable. */
    private static UnaryOperator toIRUnaryOp(UnaryOp op) {
        if (op == UnaryOp.NEGATE) return UnaryOperator.NEGATE;
        if (op == UnaryOp.ABS) return UnaryOperator.ABS;
        if (op == UnaryOp.EXP) return UnaryOperator.EXP;
        if (op == UnaryOp.LOG) return UnaryOperator.LOG;
        if (op == UnaryOp.SQRT) return UnaryOperator.SQRT;
        if (op == UnaryOp.SQUARE) return UnaryOperator.SQUARE;
        if (op == UnaryOp.SIN) return UnaryOperator.SIN;
        if (op == UnaryOp.COS) return UnaryOperator.COS;
        if (op == UnaryOp.TANH) return UnaryOperator.TANH;
        if (op == UnaryOp.RECIPROCAL) return UnaryOperator.RECIPROCAL;
        if (op == UnaryOp.LOGICAL_NOT) return UnaryOperator.LOGICAL_NOT;
        if (op == UnaryOp.BITWISE_NOT) return UnaryOperator.BITWISE_NOT;
        return null;
    }

    /**
     * Creates a broadcasted constant tensor with the given value, type, and shape. For true scalars
     * (shape.isScalar()), creates a scalar tensor.
     */
    private static Tensor broadcastedOf(Number value, DataType type, Shape shape) {
        if (shape.isScalar()) {
            if (type.isIntegral() || type == DataType.BOOL) {
                return Tensor.scalar(value.longValue(), type);
            } else {
                return Tensor.scalar(value.doubleValue(), type);
            }
        }
        // Preserve the shape by creating a broadcasted constant
        return Tensor.full(value, type, shape);
    }

    // ========== Binary Operations ==========

    static Optional<Tensor> tryFoldBinaryOp(Tensor left, Tensor right, BinaryOp op) {
        // First try ConstantComputation-based folding
        Optional<ConstantComputation> leftConst = asConstant(left);
        Optional<ConstantComputation> rightConst = asConstant(right);
        if (leftConst.isPresent() && rightConst.isPresent()) {
            ConstantComputation lc = leftConst.get();
            ConstantComputation rc = rightConst.get();
            DataType resultType =
                    TensorTypeSemantics.promoteForArithmetic(
                            lc.dataType(), rc.dataType(), op.name());
            Number result = evalBinary(lc.value(), rc.value(), resultType, op);
            // Use the larger shape to preserve broadcasting semantics
            Shape resultShape = lc.shape().size() >= rc.shape().size() ? lc.shape() : rc.shape();
            return Optional.of(broadcastedOf(result, resultType, resultShape));
        }

        // Try IR ScalarConstant-based folding
        Optional<ScalarConstant> leftSc = asScalarConstant(left);
        Optional<ScalarConstant> rightSc = asScalarConstant(right);
        if (leftSc.isPresent() && rightSc.isPresent()) {
            BinaryOperator irOp = toIRBinaryOp(op);
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
        return switch (type) {
            case DataType t when t == DataType.I8 ->
                    (byte) evalByte(a.byteValue(), b.byteValue(), op);
            case DataType t when t == DataType.I16 ->
                    (short) evalShort(a.shortValue(), b.shortValue(), op);
            case DataType t when t == DataType.I32 -> evalInt(a.intValue(), b.intValue(), op);
            case DataType t when t == DataType.I64 -> evalLong(a.longValue(), b.longValue(), op);
            case DataType t when t == DataType.FP16 || t == DataType.BF16 || t == DataType.FP32 ->
                    evalFloat(a.floatValue(), b.floatValue(), op);
            case DataType t when t == DataType.FP64 ->
                    evalDouble(a.doubleValue(), b.doubleValue(), op);
            default -> throw new UnsupportedOperationException("Unsupported type: " + type);
        };
    }

    static byte evalByte(byte a, byte b, BinaryOp op) {
        if (op == BinaryOp.ADD) return (byte) (a + b);
        if (op == BinaryOp.SUBTRACT) return (byte) (a - b);
        if (op == BinaryOp.MULTIPLY) return (byte) (a * b);
        if (op == BinaryOp.DIVIDE) return (byte) (a / b);
        if (op == BinaryOp.MIN) return (byte) Math.min(a, b);
        if (op == BinaryOp.MAX) return (byte) Math.max(a, b);
        if (op == BinaryOp.BITWISE_AND) return (byte) (a & b);
        if (op == BinaryOp.BITWISE_OR) return (byte) (a | b);
        if (op == BinaryOp.BITWISE_XOR) return (byte) (a ^ b);
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    static short evalShort(short a, short b, BinaryOp op) {
        if (op == BinaryOp.ADD) return (short) (a + b);
        if (op == BinaryOp.SUBTRACT) return (short) (a - b);
        if (op == BinaryOp.MULTIPLY) return (short) (a * b);
        if (op == BinaryOp.DIVIDE) return (short) (a / b);
        if (op == BinaryOp.MIN) return (short) Math.min(a, b);
        if (op == BinaryOp.MAX) return (short) Math.max(a, b);
        if (op == BinaryOp.BITWISE_AND) return (short) (a & b);
        if (op == BinaryOp.BITWISE_OR) return (short) (a | b);
        if (op == BinaryOp.BITWISE_XOR) return (short) (a ^ b);
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    static int evalInt(int a, int b, BinaryOp op) {
        if (op == BinaryOp.ADD) return a + b;
        if (op == BinaryOp.SUBTRACT) return a - b;
        if (op == BinaryOp.MULTIPLY) return a * b;
        if (op == BinaryOp.DIVIDE) return a / b;
        if (op == BinaryOp.MIN) return Math.min(a, b);
        if (op == BinaryOp.MAX) return Math.max(a, b);
        if (op == BinaryOp.BITWISE_AND) return a & b;
        if (op == BinaryOp.BITWISE_OR) return a | b;
        if (op == BinaryOp.BITWISE_XOR) return a ^ b;
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    static long evalLong(long a, long b, BinaryOp op) {
        if (op == BinaryOp.ADD) return a + b;
        if (op == BinaryOp.SUBTRACT) return a - b;
        if (op == BinaryOp.MULTIPLY) return a * b;
        if (op == BinaryOp.DIVIDE) return a / b;
        if (op == BinaryOp.MIN) return Math.min(a, b);
        if (op == BinaryOp.MAX) return Math.max(a, b);
        if (op == BinaryOp.BITWISE_AND) return a & b;
        if (op == BinaryOp.BITWISE_OR) return a | b;
        if (op == BinaryOp.BITWISE_XOR) return a ^ b;
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    static float evalFloat(float a, float b, BinaryOp op) {
        if (op == BinaryOp.ADD) return a + b;
        if (op == BinaryOp.SUBTRACT) return a - b;
        if (op == BinaryOp.MULTIPLY) return a * b;
        if (op == BinaryOp.DIVIDE) return a / b;
        if (op == BinaryOp.MIN) return Math.min(a, b);
        if (op == BinaryOp.MAX) return Math.max(a, b);
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    static double evalDouble(double a, double b, BinaryOp op) {
        if (op == BinaryOp.ADD) return a + b;
        if (op == BinaryOp.SUBTRACT) return a - b;
        if (op == BinaryOp.MULTIPLY) return a * b;
        if (op == BinaryOp.DIVIDE) return a / b;
        if (op == BinaryOp.MIN) return Math.min(a, b);
        if (op == BinaryOp.MAX) return Math.max(a, b);
        throw new UnsupportedOperationException("Cannot fold: " + op);
    }

    // ========== Unary Operations ==========

    static Optional<Tensor> tryFoldUnaryOp(Tensor tensor, UnaryOp op) {
        // First try ConstantComputation-based folding
        Optional<ConstantComputation> constant = asConstant(tensor);
        if (constant.isPresent()) {
            ConstantComputation c = constant.get();
            try {
                Number result = evalUnary(c.value(), c.dataType(), op);
                // Preserve the original shape
                return Optional.of(broadcastedOf(result, c.dataType(), c.shape()));
            } catch (UnsupportedOperationException e) {
                // Fall through to try IR folding
            }
        }

        // Try IR ScalarConstant-based folding
        Optional<ScalarConstant> sc = asScalarConstant(tensor);
        if (sc.isPresent()) {
            UnaryOperator irOp = toIRUnaryOp(op);
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
        return switch (type) {
            case DataType t when t == DataType.I8 -> evalByte(a.byteValue(), op);
            case DataType t when t == DataType.I16 -> evalShort(a.shortValue(), op);
            case DataType t when t == DataType.I32 -> evalInt(a.intValue(), op);
            case DataType t when t == DataType.I64 -> evalLong(a.longValue(), op);
            case DataType t when t == DataType.FP16 || t == DataType.BF16 || t == DataType.FP32 ->
                    evalFloat(a.floatValue(), op);
            case DataType t when t == DataType.FP64 -> evalDouble(a.doubleValue(), op);
            case DataType t when t == DataType.BOOL -> (long) evalBool(a.longValue() != 0, op);
            default -> throw new UnsupportedOperationException("Unsupported type: " + type);
        };
    }

    static byte evalByte(byte a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return (byte) -a;
        if (op == UnaryOp.ABS) return (byte) Math.abs(a);
        if (op == UnaryOp.BITWISE_NOT) return (byte) ~a;
        throw new UnsupportedOperationException("Unsupported unary op for I8: " + op);
    }

    static short evalShort(short a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return (short) -a;
        if (op == UnaryOp.ABS) return (short) Math.abs(a);
        if (op == UnaryOp.BITWISE_NOT) return (short) ~a;
        throw new UnsupportedOperationException("Unsupported unary op for I16: " + op);
    }

    static int evalInt(int a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return -a;
        if (op == UnaryOp.ABS) return Math.abs(a);
        if (op == UnaryOp.BITWISE_NOT) return ~a;
        throw new UnsupportedOperationException("Unsupported unary op for I32: " + op);
    }

    static long evalLong(long a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return -a;
        if (op == UnaryOp.ABS) return Math.abs(a);
        if (op == UnaryOp.BITWISE_NOT) return ~a;
        throw new UnsupportedOperationException("Unsupported unary op for I64: " + op);
    }

    static float evalFloat(float a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return -a;
        if (op == UnaryOp.ABS) return Math.abs(a);
        if (op == UnaryOp.EXP) return (float) Math.exp(a);
        if (op == UnaryOp.LOG) return (float) Math.log(a);
        if (op == UnaryOp.SQRT) return (float) Math.sqrt(a);
        if (op == UnaryOp.SIN) return (float) Math.sin(a);
        if (op == UnaryOp.COS) return (float) Math.cos(a);
        if (op == UnaryOp.TANH) return (float) Math.tanh(a);
        if (op == UnaryOp.RECIPROCAL) return 1.0f / a;
        throw new UnsupportedOperationException("Unsupported unary op for FP32: " + op);
    }

    static double evalDouble(double a, UnaryOp op) {
        if (op == UnaryOp.NEGATE) return -a;
        if (op == UnaryOp.ABS) return Math.abs(a);
        if (op == UnaryOp.EXP) return Math.exp(a);
        if (op == UnaryOp.LOG) return Math.log(a);
        if (op == UnaryOp.SQRT) return Math.sqrt(a);
        if (op == UnaryOp.SIN) return Math.sin(a);
        if (op == UnaryOp.COS) return Math.cos(a);
        if (op == UnaryOp.TANH) return Math.tanh(a);
        if (op == UnaryOp.RECIPROCAL) return 1.0 / a;
        throw new UnsupportedOperationException("Unsupported unary op for FP64: " + op);
    }

    static byte evalBool(boolean a, UnaryOp op) {
        if (op == UnaryOp.LOGICAL_NOT) return (byte) (a ? 0 : 1);
        throw new UnsupportedOperationException("Unsupported unary op for BOOL: " + op);
    }

    // ========== Comparison Operations ==========

    static Optional<Tensor> tryFoldCompareOp(Tensor left, Tensor right, BinaryOp op) {
        Optional<ConstantComputation> leftConst = asConstant(left);
        Optional<ConstantComputation> rightConst = asConstant(right);
        if (leftConst.isEmpty() || rightConst.isEmpty()) {
            return Optional.empty();
        }
        ConstantComputation lc = leftConst.get();
        ConstantComputation rc = rightConst.get();
        DataType commonType =
                TensorTypeSemantics.promoteForComparison(
                        lc.dataType(), rc.dataType(), op.name());
        boolean result = evalCompare(lc.value(), rc.value(), commonType, op);
        return Optional.of(Tensor.scalar(result ? 1L : 0L, DataType.BOOL));
    }

    static boolean evalCompare(Number a, Number b, DataType type, BinaryOp op) {
        return switch (type) {
            case DataType t when t == DataType.BOOL ->
                    evalCompare((byte) (a.longValue() != 0 ? 1 : 0), (byte) (b.longValue() != 0 ? 1 : 0), op);
            case DataType t when t == DataType.I8 -> evalCompare(a.byteValue(), b.byteValue(), op);
            case DataType t when t == DataType.I16 ->
                    evalCompare(a.shortValue(), b.shortValue(), op);
            case DataType t when t == DataType.I32 -> evalCompare(a.intValue(), b.intValue(), op);
            case DataType t when t == DataType.I64 -> evalCompare(a.longValue(), b.longValue(), op);
            case DataType t when t == DataType.FP16 || t == DataType.BF16 || t == DataType.FP32 ->
                    evalCompare(a.floatValue(), b.floatValue(), op);
            case DataType t when t == DataType.FP64 ->
                    evalCompare(a.doubleValue(), b.doubleValue(), op);
            default -> throw new UnsupportedOperationException("Unsupported type: " + type);
        };
    }

    static boolean evalCompare(byte a, byte b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    static boolean evalCompare(short a, short b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    static boolean evalCompare(int a, int b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    static boolean evalCompare(long a, long b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    static boolean evalCompare(float a, float b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    static boolean evalCompare(double a, double b, BinaryOp op) {
        if (op == BinaryOp.EQUAL) return a == b;
        if (op == BinaryOp.LESS_THAN) return a < b;
        throw new UnsupportedOperationException("Cannot fold compare: " + op);
    }

    // ========== Cast Operation ==========

    static Optional<Tensor> tryFoldCast(Tensor tensor, DataType targetType) {
        // First try ConstantComputation-based folding
        Optional<ConstantComputation> constant = asConstant(tensor);
        if (constant.isPresent()) {
            ConstantComputation c = constant.get();
            Number result = cast(c.value(), targetType);
            // Preserve the original shape
            return Optional.of(broadcastedOf(result, targetType, c.shape()));
        }

        // Try IR ScalarConstant-based folding
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
        return switch (targetType) {
            case DataType t when t == DataType.I8 -> value.byteValue();
            case DataType t when t == DataType.I16 -> value.shortValue();
            case DataType t when t == DataType.I32 -> value.intValue();
            case DataType t when t == DataType.I64 -> value.longValue();
            case DataType t when t == DataType.FP16 || t == DataType.BF16 || t == DataType.FP32 ->
                    value.floatValue();
            case DataType t when t == DataType.FP64 -> value.doubleValue();
            case DataType t when t == DataType.BOOL -> value.longValue() != 0 ? 1L : 0L;
            default ->
                    throw new UnsupportedOperationException(
                            "Unsupported target type: " + targetType);
        };
    }
}
