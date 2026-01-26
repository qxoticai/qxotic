package ai.qxotic.jota.tensor;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.impl.ViewTransforms;
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

    //    Tensor add(Tensor a, Number scalar);
    //
    //    Tensor subtract(Tensor a, Number scalar);
    //
    //    Tensor multiply(Tensor a, Number scalar);
    //
    //    Tensor divide(Tensor a, Number scalar);

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

    Tensor to(Tensor x, Device device);

    Tensor contiguous(Tensor x);

    // === Bitwise Operations ===

    Tensor bitwiseNot(Tensor x); // ~x

    Tensor bitwiseAnd(Tensor a, Tensor b); // a & b

    Tensor bitwiseOr(Tensor a, Tensor b); // a | b

    Tensor bitwiseXor(Tensor a, Tensor b); // a ^ b

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

    // === Linear Algebra ===

    Tensor matmul(Tensor a, Tensor b);

    Tensor batchedMatmul(Tensor a, Tensor b);

    // === Shape Operations ===

    Tensor viewTransform(Tensor x, Layout layout, long byteOffsetDelta, String hint);

    default Tensor transpose(Tensor x, int axis0, int axis1) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.transpose(x.layout(), axis0, axis1);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "transpose");
    }

    default Tensor transpose(Tensor x) {
        return transpose(x, -2, -1);
    }

    Tensor reshape(Tensor x, Shape newShape);

    default Tensor view(Tensor x, Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(x.layout(), newShape);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "view");
    }

    default Tensor broadcast(Tensor x, Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.broadcast(x.layout(), targetShape);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "broadcast");
    }

    default Tensor expand(Tensor x, Shape targetShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.expand(x.layout(), targetShape);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "expand");
    }

    default Tensor permute(Tensor x, int... permutationIndices) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.permute(x.layout(), permutationIndices);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "permute");
    }

    default Tensor slice(Tensor x, int axis, long start, long end) {
        return slice(x, axis, start, end, 1);
    }

    default Tensor slice(Tensor x, int axis, long start, long end, long indexStride) {
        ViewTransforms.ViewTransformSpec spec =
                ViewTransforms.slice(x.layout(), x.dataType(), axis, start, end, indexStride);
        return viewTransform(x, spec.layout(), spec.byteOffsetDelta(), "slice");
    }

    // === Type Conversion ===

    Tensor cast(Tensor x, DataType targetType);

    //    Tensor quantize(Tensor x, DataType quantType);
    //
    //    Tensor dequantize(Tensor x, DataType targetType);
}
