package com.qxotic.jota.runtime;

import com.qxotic.jota.DataType;
import com.qxotic.jota.tensor.BinaryOp;
import com.qxotic.jota.tensor.Tensor;
import com.qxotic.jota.tensor.Tracer;
import com.qxotic.jota.tensor.UnaryOp;
import java.util.List;

public final class TracingEagerKernels implements DeviceRuntime.EagerKernels {

    @Override
    public Tensor unary(UnaryOp op, Tensor x) {
        return Tracer.trace(x, t -> applyUnary(op, t));
    }

    @Override
    public Tensor binary(BinaryOp op, Tensor a, Tensor b) {
        return Tracer.trace(a, b, (left, right) -> applyBinary(op, left, right));
    }

    @Override
    public Tensor compare(BinaryOp op, Tensor a, Tensor b) {
        return Tracer.trace(a, b, (left, right) -> applyCompare(op, left, right));
    }

    @Override
    public Tensor logical(BinaryOp op, Tensor a, Tensor b) {
        return Tracer.trace(a, b, (left, right) -> applyLogical(op, left, right));
    }

    @Override
    public Tensor cast(Tensor x, DataType targetType) {
        return Tracer.trace(x, t -> t.cast(targetType));
    }

    @Override
    public Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue) {
        return Tracer.trace(
                List.of(condition, trueValue, falseValue),
                tensors -> tensors.get(0).where(tensors.get(1), tensors.get(2)));
    }

    @Override
    public Tensor reduce(
            DeviceRuntime.ReductionOp op,
            Tensor x,
            DataType accumulatorType,
            boolean keepDims,
            int axis,
            int... axes) {
        return Tracer.trace(
                x,
                t ->
                        switch (op) {
                            case SUM -> t.sum(accumulatorType, keepDims, axis, axes);
                            case PRODUCT -> t.product(accumulatorType, keepDims, axis, axes);
                            case MIN -> t.min(keepDims, axis, axes);
                            case MAX -> t.max(keepDims, axis, axes);
                        });
    }

    @Override
    public Tensor matmul(Tensor a, Tensor b) {
        return Tracer.trace(a, b, Tensor::matmul);
    }

    @Override
    public Tensor batchedMatmul(Tensor a, Tensor b) {
        return Tracer.trace(a, b, Tensor::batchedMatmul);
    }

    @Override
    public Tensor gather(Tensor input, Tensor indices, int axis) {
        return Tracer.trace(input, indices, (table, idx) -> table.gather(idx, axis));
    }

    private static Tensor applyUnary(UnaryOp op, Tensor tensor) {
        return switch (op) {
            case IDENTITY -> tensor;
            case CAST ->
                    throw new UnsupportedOperationException(
                            "CAST unary op requires explicit target type");
            case NEGATE -> tensor.negate();
            case ABS -> tensor.abs();
            case EXP -> tensor.exp();
            case LOG -> tensor.log();
            case SQRT -> tensor.sqrt();
            case SQUARE -> tensor.square();
            case SIN -> tensor.sin();
            case COS -> tensor.cos();
            case TAN -> throw new UnsupportedOperationException("TAN unary op not supported");
            case TANH -> tensor.tanh();
            case RECIPROCAL -> tensor.reciprocal();
            case LOGICAL_NOT -> tensor.logicalNot();
            case BITWISE_NOT -> tensor.bitwiseNot();
        };
    }

    private static Tensor applyBinary(BinaryOp op, Tensor left, Tensor right) {
        return switch (op) {
            case ADD -> left.add(right);
            case SUBTRACT -> left.subtract(right);
            case MULTIPLY -> left.multiply(right);
            case DIVIDE -> left.divide(right);
            case MIN -> left.min(right);
            case MAX -> left.max(right);
            case POW -> throw new UnsupportedOperationException("POW binary op not supported");
            case BITWISE_AND -> left.bitwiseAnd(right);
            case BITWISE_OR -> left.bitwiseOr(right);
            case BITWISE_XOR -> left.bitwiseXor(right);
            case LEFT_SHIFT -> left.leftShift(right);
            case RIGHT_SHIFT -> left.rightShift(right);
            case RIGHT_SHIFT_UNSIGNED -> left.rightShiftUnsigned(right);
            case LOGICAL_AND, LOGICAL_OR, LOGICAL_XOR, EQUAL, LESS_THAN ->
                    throw new UnsupportedOperationException(
                            "Binary op not supported in this method: " + op);
        };
    }

    private static Tensor applyCompare(BinaryOp op, Tensor left, Tensor right) {
        return switch (op) {
            case EQUAL -> left.equal(right);
            case LESS_THAN -> left.lessThan(right);
            default ->
                    throw new UnsupportedOperationException("Comparison op not supported: " + op);
        };
    }

    private static Tensor applyLogical(BinaryOp op, Tensor left, Tensor right) {
        return switch (op) {
            case LOGICAL_AND -> left.logicalAnd(right);
            case LOGICAL_OR -> left.logicalOr(right);
            case LOGICAL_XOR -> left.logicalXor(right);
            default -> throw new UnsupportedOperationException("Logical op not supported: " + op);
        };
    }
}
