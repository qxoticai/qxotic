package ai.qxotic.jota.ir.irt;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/** Reduction operation node in IR-T. */
public record ReductionOp(ReductionOperator op, IRTNode input, int[] axes, boolean keepDims)
        implements IRTNode {

    public ReductionOp {
        Objects.requireNonNull(op);
        Objects.requireNonNull(input);
        Objects.requireNonNull(axes);
        if (axes.length == 0) {
            throw new IllegalArgumentException("axes cannot be empty");
        }
        Set<Integer> uniqueAxes = new HashSet<>();
        for (int axis : axes) {
            if (axis < 0 || axis >= input.layout().shape().rank()) {
                throw new IllegalArgumentException(
                        "axis "
                                + axis
                                + " out of bounds for tensor with rank "
                                + input.layout().shape().rank());
            }
            if (!uniqueAxes.add(axis)) {
                throw new IllegalArgumentException("axes must be unique, found duplicate: " + axis);
            }
        }
    }

    @Override
    public DataType dataType() {
        return input.dataType();
    }

    @Override
    public Layout layout() {
        Shape inputShape = input.layout().shape();
        int inputRank = inputShape.rank();

        if (keepDims) {
            long[] reducedDims = new long[inputRank];
            for (int i = 0; i < inputRank; i++) {
                if (isReduced(i)) {
                    reducedDims[i] = 1;
                } else {
                    reducedDims[i] = inputShape.flatAt(i);
                }
            }
            return Layout.of(
                    Shape.flat(reducedDims),
                    ai.qxotic.jota.Stride.rowMajor(Shape.flat(reducedDims)));
        } else {
            long[] reducedDims = new long[inputRank - axes.length];
            int idx = 0;
            for (int i = 0; i < inputRank; i++) {
                if (!isReduced(i)) {
                    reducedDims[idx++] = inputShape.flatAt(i);
                }
            }
            return Layout.of(
                    Shape.flat(reducedDims),
                    ai.qxotic.jota.Stride.rowMajor(Shape.flat(reducedDims)));
        }
    }

    private boolean isReduced(int axis) {
        for (int a : axes) {
            if (a == axis) {
                return true;
            }
        }
        return false;
    }
}
