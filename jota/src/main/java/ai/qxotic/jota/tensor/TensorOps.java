package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.memory.MemoryContext;
import java.util.Arrays;
import java.util.stream.IntStream;

public interface TensorOps {

    MemoryContext<?> context();

    // === Elementwise - Binary ===

    Tensor add(Tensor a, Tensor b);

    Tensor subtract(Tensor a, Tensor b);

    Tensor multiply(Tensor a, Tensor b);

    Tensor divide(Tensor a, Tensor b);

    Tensor min(Tensor a, Tensor b);

    Tensor max(Tensor a, Tensor b);

    // === Elementwise - Scalar ===

    Tensor add(Tensor a, Number scalar);

    Tensor subtract(Tensor a, Number scalar);

    Tensor multiply(Tensor a, Number scalar);

    Tensor divide(Tensor a, Number scalar);

    // === Elementwise - Unary ===

    Tensor negate(Tensor x);

    Tensor abs(Tensor x);

    Tensor exp(Tensor x);

    Tensor log(Tensor x);

    Tensor sqrt(Tensor x);

    Tensor square(Tensor x);

    Tensor sin(Tensor x);

    Tensor cos(Tensor x);

    Tensor tanh(Tensor x);

    Tensor sigmoid(Tensor x);

    Tensor relu(Tensor x);

    Tensor gelu(Tensor x);

    Tensor silu(Tensor x);

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

    Tensor mean(Tensor x, int axis, boolean keepDims);

    default Tensor mean(Tensor x, int axis) {
        return mean(x, axis, false);
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

    Tensor meanAll(Tensor x);

    Tensor maxAll(Tensor x);

    Tensor minAll(Tensor x);

    // === Linear Algebra ===

    Tensor matmul(Tensor a, Tensor b);

    Tensor batchedMatmul(Tensor a, Tensor b);

    // === Shape Operations ===

    Tensor transpose(Tensor x, int axis0, int axis1);

    default Tensor transpose(Tensor x) {
        return transpose(x, -2, -1);
    }

    Tensor reshape(Tensor x, Shape newShape);

    Tensor view(Tensor x, Shape newShape);

    Tensor broadcast(Tensor x, Shape targetShape);

    Tensor expand(Tensor x, Shape targetShape);

    Tensor slice(Tensor x, int axis, long start, long end);

    // === Special Operations ===

    Tensor softmax(Tensor x, int axis);

    Tensor layerNorm(Tensor x, Tensor weight, Tensor bias, float eps);

    Tensor rmsNorm(Tensor x, Tensor weight, float eps);

    // === Type Conversion ===

    Tensor cast(Tensor x, DataType targetType);

    Tensor quantize(Tensor x, DataType quantType);

    Tensor dequantize(Tensor x, DataType targetType);

    // === Memory Operations ===

    Tensor copy(Tensor x);

    Tensor clone(Tensor x);

    void copyInto(Tensor src, Tensor dst);
}
