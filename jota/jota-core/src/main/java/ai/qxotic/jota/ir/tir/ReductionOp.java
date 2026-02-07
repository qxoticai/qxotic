package ai.qxotic.jota.ir.tir;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Shape;
import java.util.Objects;

/** Reduction operation node in IR-T. */
public record ReductionOp(
        ReductionOperator op,
        TIRNode input,
        int[] axes,
        boolean keepDims,
        DataType accumulatorType,
        Shape shape)
        implements TIRNode {

    public ReductionOp(
            ReductionOperator op,
            TIRNode input,
            int[] axes,
            boolean keepDims,
            DataType accumulatorType) {
        this(
                op,
                input,
                axes,
                keepDims,
                accumulatorType,
                reduceShape(input.shape(), axes, keepDims));
    }

    public ReductionOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(input);
        Objects.requireNonNull(axes);
        Objects.requireNonNull(accumulatorType);
        Objects.requireNonNull(shape);
    }

    @Override
    public DataType dataType() {
        return accumulatorType;
    }

    @Override
    public Shape shape() {
        return shape;
    }

    private static Shape reduceShape(Shape inputShape, int[] axes, boolean keepDims) {
        int rank = inputShape.rank();
        boolean[] reduced = new boolean[rank];
        for (int axis : axes) {
            if (axis >= 0 && axis < rank) {
                reduced[axis] = true;
            }
        }
        if (keepDims) {
            long[] dims = new long[rank];
            for (int i = 0; i < rank; i++) {
                dims[i] = reduced[i] ? 1 : inputShape.flatAt(i);
            }
            return Shape.flat(dims);
        }
        long[] dims = new long[rank - axes.length];
        int idx = 0;
        for (int i = 0; i < rank; i++) {
            if (!reduced[i]) {
                dims[idx++] = inputShape.flatAt(i);
            }
        }
        return Shape.flat(dims);
    }
}
