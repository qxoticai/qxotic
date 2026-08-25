package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Util;
import com.qxotic.jota.ir.tir.ViewOperation;
import com.qxotic.jota.memory.impl.ViewTransforms;
import java.util.Arrays;
import java.util.stream.IntStream;

interface TensorOps {

    // === Elementwise - Binary ===

    Tensor add(Tensor a, Tensor b);

    Tensor subtract(Tensor a, Tensor b);

    Tensor multiply(Tensor a, Tensor b);

    Tensor divide(Tensor a, Tensor b);

    Tensor min(Tensor a, Tensor b);

    Tensor max(Tensor a, Tensor b);

    // === Elementwise - Unary ===

    Tensor negate(Tensor x);

    Tensor abs(Tensor x);

    Tensor exp(Tensor x);

    Tensor log(Tensor x);

    Tensor sqrt(Tensor x);

    Tensor sin(Tensor x);

    Tensor cos(Tensor x);

    Tensor tanh(Tensor x);

    Tensor reciprocal(Tensor x);

    // === Bitwise Operations ===

    Tensor bitwiseNot(Tensor x); // ~x

    Tensor bitwiseAnd(Tensor a, Tensor b); // a & b

    Tensor bitwiseOr(Tensor a, Tensor b); // a | b

    Tensor bitwiseXor(Tensor a, Tensor b); // a ^ b

    Tensor leftShift(Tensor a, Tensor b); // a << b

    Tensor rightShift(Tensor a, Tensor b); // a >> b

    Tensor rightShiftUnsigned(Tensor a, Tensor b); // a >>> b

    // === Boolean Operations ===

    Tensor logicalNot(Tensor x); // !x

    Tensor logicalAnd(Tensor a, Tensor b); // a && b

    Tensor logicalOr(Tensor a, Tensor b); // a || b

    Tensor logicalXor(Tensor a, Tensor b); // a ^ b

    Tensor equal(Tensor a, Tensor b);

    Tensor lessThan(Tensor a, Tensor b);

    default Tensor notEqual(Tensor a, Tensor b) {
        return logicalNot(equal(a, b));
    }

    default Tensor greaterThan(Tensor a, Tensor b) {
        return lessThan(b, a);
    }

    default Tensor lessThanOrEqual(Tensor a, Tensor b) {
        return logicalNot(lessThan(b, a));
    }

    default Tensor greaterThanOrEqual(Tensor a, Tensor b) {
        return logicalNot(lessThan(a, b));
    }

    Tensor where(Tensor condition, Tensor trueValue, Tensor falseValue);

    // === Reduction Operations ===

    Tensor sum(Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes);

    default Tensor sum(Tensor x, DataType accumulatorType, int _axis, int... _axes) {
        return sum(x, accumulatorType, false, _axis, _axes);
    }

    default Tensor sum(Tensor x, DataType accumulatorType) {
        int rank = x.shape().rank();
        if (rank == 0) {
            return x;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return sum(x, accumulatorType, false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    Tensor product(Tensor x, DataType accumulatorType, boolean keepDims, int _axis, int... _axes);

    default Tensor product(Tensor x, DataType accumulatorType, int _axis, int... _axes) {
        return product(x, accumulatorType, false, _axis, _axes);
    }

    default Tensor product(Tensor x, DataType accumulatorType) {
        int rank = x.shape().rank();
        if (rank == 0) {
            return x;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return product(
                x, accumulatorType, false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    Tensor max(Tensor x, boolean keepDims, int _axis, int... _axes);

    default Tensor max(Tensor x, int _axis, int... _axes) {
        return max(x, false, _axis, _axes);
    }

    default Tensor max(Tensor x) {
        int rank = x.shape().rank();
        if (rank == 0) {
            return x;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return max(x, false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    Tensor min(Tensor x, boolean keepDims, int _axis, int... _axes);

    default Tensor min(Tensor x, int _axis, int... _axes) {
        return min(x, false, _axis, _axes);
    }

    default Tensor min(Tensor x) {
        int rank = x.shape().rank();
        if (rank == 0) {
            return x;
        }
        int[] axes = IntStream.range(0, rank).toArray();
        return min(x, false, axes[0], Arrays.copyOfRange(axes, 1, axes.length));
    }

    // === Indexing Operations ===

    /**
     * Gathers elements from the input tensor along the specified axis according to the indices.
     *
     * <p>For embedding lookup (the most common use case): - input: [vocabSize, hiddenSize] tensor
     * containing embeddings - indices: [batchSize, seqLen] tensor containing token IDs - axis: 0 -
     * output: [batchSize, seqLen, hiddenSize] tensor with gathered embeddings
     *
     * @param input the source tensor to gather from
     * @param indices the indices tensor (must be integral type)
     * @param _axis the axis along which to gather (wrap-around semantics)
     * @return the gathered tensor
     */
    Tensor gather(Tensor input, Tensor indices, int _axis);

    // === Linear Algebra ===

    Tensor dot(Tensor a, Tensor b, DataType accumulatorType);

    default Tensor dot(Tensor a, Tensor b) {
        DataType inputType = a.dataType();
        if (!inputType.isFloatingPoint() || !b.dataType().isFloatingPoint()) {
            throw new IllegalArgumentException(
                    "dot(a, b) is floating-point only; use dot(a, b, accumulatorType) for integral"
                            + " inputs");
        }
        return dot(a, b, inputType);
    }

    Tensor matmul(Tensor a, Tensor b);

    Tensor batchedMatmul(Tensor a, Tensor b);

    // === Shape Operations ===

    Tensor viewTransform(Tensor x, ViewOperation operation, ViewTransforms.Result result);

    default Tensor transpose(Tensor x, int _axis0, int _axis1) {
        int rank = x.shape().rank();
        int axis0 = Util.wrapAround(_axis0, rank);
        int axis1 = Util.wrapAround(_axis1, rank);
        int[] permutation = IntStream.range(0, rank).toArray();
        permutation[axis0] = axis1;
        permutation[axis1] = axis0;
        return permute(x, permutation);
    }

    default Tensor transpose(Tensor x) {
        return transpose(x, -2, -1);
    }

    Tensor reshape(Tensor x, Shape newShape);

    default Tensor view(Tensor x, Shape newShape) {
        ViewTransforms.Result result = ViewTransforms.view(x.layout(), newShape);
        return viewTransform(x, new ViewOperation.Reshape(x.shape(), newShape), result);
    }

    default Tensor unsqueeze(Tensor x, int axis) {
        ViewTransforms.Result result = ViewTransforms.unsqueeze(x.layout(), axis);
        return viewTransform(
                x, new ViewOperation.Reshape(x.shape(), result.layout().shape()), result);
    }

    default Tensor broadcast(Tensor x, Shape targetShape) {
        ViewTransforms.Result result = ViewTransforms.broadcast(x.layout(), targetShape);
        return viewTransform(x, new ViewOperation.Broadcast(x.shape(), targetShape), result);
    }

    default Tensor expand(Tensor x, Shape targetShape) {
        ViewTransforms.Result result = ViewTransforms.expand(x.layout(), targetShape);
        return viewTransform(x, new ViewOperation.Expand(x.shape(), targetShape), result);
    }

    default Tensor permute(Tensor x, int... permutationIndices) {
        ViewTransforms.Result result = ViewTransforms.permute(x.layout(), permutationIndices);
        return viewTransform(x, new ViewOperation.Transpose(permutationIndices), result);
    }

    default Tensor slice(Tensor x, int _axis, long start, long end) {
        return slice(x, _axis, start, end, 1);
    }

    default Tensor slice(Tensor x, int _axis, long start, long end, long indexStride) {
        int axis = Util.wrapAround(_axis, x.shape().rank());
        ViewTransforms.Result result =
                ViewTransforms.slice(x.layout(), x.dataType(), axis, start, end, indexStride);
        return viewTransform(x, new ViewOperation.Slice(axis, start, indexStride), result);
    }

    // === Type Conversion ===

    Tensor cast(Tensor x, DataType targetType);

    // === Misc Conversion ===

    Tensor to(Tensor x, Device device);

    Tensor contiguous(Tensor x);
}
