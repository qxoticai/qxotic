package ai.qxotic.jota.tensor;

import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Util;

final class TensorSemantics {

    private TensorSemantics() {}

    static int normalizeAxis(int rank, int axis) {
        return Util.wrapAround(axis, rank);
    }

    static int[] normalizeReductionAxes(int rank, int firstAxis, int... otherAxes) {
        int[] raw = new int[otherAxes.length + 1];
        raw[0] = firstAxis;
        System.arraycopy(otherAxes, 0, raw, 1, otherAxes.length);

        boolean[] seen = new boolean[rank];
        int uniqueCount = 0;
        for (int axis : raw) {
            int normalized = normalizeAxis(rank, axis);
            if (!seen[normalized]) {
                seen[normalized] = true;
                uniqueCount++;
            }
        }

        int[] normalizedAxes = new int[uniqueCount];
        int index = 0;
        for (int i = 0; i < rank; i++) {
            if (seen[i]) {
                normalizedAxes[index++] = i;
            }
        }
        return normalizedAxes;
    }

    static Shape reduceShape(Shape inputShape, int[] axes, boolean keepDims) {
        boolean[] reduced = reductionMask(inputShape.rank(), axes);
        if (keepDims) {
            long[] dims = new long[inputShape.rank()];
            for (int i = 0; i < inputShape.rank(); i++) {
                dims[i] = reduced[i] ? 1 : inputShape.flatAt(i);
            }
            return Shape.flat(dims);
        }

        long[] dims = new long[inputShape.rank() - axes.length];
        int idx = 0;
        for (int i = 0; i < inputShape.rank(); i++) {
            if (!reduced[i]) {
                dims[idx++] = inputShape.flatAt(i);
            }
        }
        return Shape.flat(dims);
    }

    static boolean[] reductionMask(int rank, int[] axes) {
        boolean[] reduced = new boolean[rank];
        for (int axis : axes) {
            reduced[axis] = true;
        }
        return reduced;
    }

    static long[] projectReducedCoord(long[] inputCoord, boolean[] reducedMask, boolean keepDims) {
        if (keepDims) {
            long[] outputCoord = new long[inputCoord.length];
            for (int i = 0; i < inputCoord.length; i++) {
                outputCoord[i] = reducedMask[i] ? 0L : inputCoord[i];
            }
            return outputCoord;
        }

        int outputRank = 0;
        for (int i = 0; i < inputCoord.length; i++) {
            if (!reducedMask[i]) {
                outputRank++;
            }
        }

        long[] outputCoord = new long[outputRank];
        int idx = 0;
        for (int i = 0; i < inputCoord.length; i++) {
            if (!reducedMask[i]) {
                outputCoord[idx++] = inputCoord[i];
            }
        }
        return outputCoord;
    }

    static Shape requireCompatibleShape(Shape left, Shape right) {
        boolean leftIsTrueScalar = left.isScalar();
        boolean rightIsTrueScalar = right.isScalar();
        if (leftIsTrueScalar && rightIsTrueScalar) {
            return left;
        }
        if (leftIsTrueScalar && !rightIsTrueScalar) {
            return right;
        }
        if (!leftIsTrueScalar && rightIsTrueScalar) {
            return left;
        }
        if (!left.isCongruentWith(right)) {
            throw new IllegalArgumentException(
                    "Incompatible shapes: "
                            + left
                            + " vs "
                            + right
                            + ". Note: Only true scalar tensors (shape.isScalar() == true) can be broadcast, "
                            + "not broadcasted tensors. Use Tensor.scalar(value) to create scalar values.");
        }
        return left;
    }

    static Shape resolveWhereShape(Shape conditionShape, Shape trueShape, Shape falseShape) {
        Shape valueShape = requireCompatibleShape(trueShape, falseShape);
        if (conditionShape.isScalar()) {
            return valueShape;
        }
        if (valueShape.isScalar()) {
            return conditionShape;
        }
        if (!conditionShape.isCongruentWith(valueShape)) {
            throw new IllegalArgumentException(
                    "Incompatible shapes in where(): condition shape "
                            + conditionShape
                            + " is not compatible with value shapes "
                            + valueShape
                            + ". Only true scalar tensors can be broadcast.");
        }
        return valueShape;
    }
}
