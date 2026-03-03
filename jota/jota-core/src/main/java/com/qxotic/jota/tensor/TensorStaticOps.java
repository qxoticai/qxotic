package com.qxotic.jota.tensor;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Arrays;
import java.util.Objects;

final class TensorStaticOps {

    private TensorStaticOps() {}

    static Tensor concat(int _axis, Tensor first, Tensor second, Tensor... rest) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(rest, "rest");

        Shape firstShape = first.shape().flattenModes();
        int rank = firstShape.rank();
        int axis = normalizeAxis(rank, _axis);

        Tensor acc = asModeTensor(first, firstShape);
        Shape secondModeShape = second.shape().flattenModes();
        acc = concatPair(axis, acc, asModeTensor(second, secondModeShape));
        for (Tensor next : rest) {
            Objects.requireNonNull(next, "concat tensor");
            Shape nextModeShape = next.shape().flattenModes();
            acc = concatPair(axis, acc, asModeTensor(next, nextModeShape));
        }
        return acc;
    }

    static Tensor stack(int _axis, Tensor first, Tensor second, Tensor... rest) {
        Objects.requireNonNull(first, "first");
        Objects.requireNonNull(second, "second");
        Objects.requireNonNull(rest, "rest");

        Shape firstShape = first.shape().flattenModes();
        int axis = normalizeAxis(firstShape.rank() + 1, _axis);
        Shape firstExpanded = insertAxisShape(firstShape, axis, 1L);

        Tensor[] expanded = new Tensor[2 + rest.length];
        expanded[0] = asModeTensor(first, firstShape).view(firstExpanded);
        Shape secondShape = second.shape().flattenModes();
        expanded[1] =
                asModeTensor(second, secondShape).view(insertAxisShape(secondShape, axis, 1L));
        for (int i = 0; i < rest.length; i++) {
            Tensor next = Objects.requireNonNull(rest[i], "stack tensor");
            Shape nextShape = next.shape().flattenModes();
            expanded[i + 2] =
                    asModeTensor(next, nextShape).view(insertAxisShape(nextShape, axis, 1L));
        }

        Tensor acc = concat(axis, expanded[0], expanded[1]);
        for (int i = 2; i < expanded.length; i++) {
            acc = concat(axis, acc, expanded[i]);
        }
        return acc;
    }

    static Tensor[] split(
            int _axis, Tensor input, long firstSize, long secondSize, long... restSizes) {
        Objects.requireNonNull(input, "input");
        Objects.requireNonNull(restSizes, "restSizes");

        Shape shape = input.shape();
        int axis = normalizeAxis(shape.rank(), _axis);
        if (shape.modeAt(axis).rank() != 1) {
            throw new IllegalArgumentException(
                    "split axis cannot be nested: axis=" + _axis + ", mode=" + shape.modeAt(axis));
        }

        long[] sizes = new long[2 + restSizes.length];
        sizes[0] = firstSize;
        sizes[1] = secondSize;
        System.arraycopy(restSizes, 0, sizes, 2, restSizes.length);

        long axisSize = shape.size(axis);
        long[] resolved = resolveSplitSizes(axisSize, sizes);

        Tensor[] out = new Tensor[resolved.length];
        long start = 0;
        for (int i = 0; i < resolved.length; i++) {
            long size = resolved[i];
            out[i] = input.slice(axis, start, start + size);
            start += size;
        }
        return out;
    }

    private static int normalizeAxis(int rank, int axis) {
        return TensorSemantics.normalizeAxis(rank, axis);
    }

    private static Tensor asModeTensor(Tensor tensor, Shape modeShape) {
        return tensor.shape().equals(modeShape) ? tensor : tensor.view(modeShape);
    }

    private static Shape insertAxisShape(Shape base, int axis, long insertedSize) {
        long[] dims = base.toArray();
        long[] out = new long[dims.length + 1];
        for (int i = 0, j = 0; i < out.length; i++) {
            if (i == axis) {
                out[i] = insertedSize;
            } else {
                out[i] = dims[j++];
            }
        }
        return Shape.flat(out);
    }

    private static long[] resolveSplitSizes(long axisSize, long[] sizes) {
        int inferIndex = -1;
        long knownSum = 0;
        for (int i = 0; i < sizes.length; i++) {
            long s = sizes[i];
            if (s == -1) {
                if (inferIndex >= 0) {
                    throw new IllegalArgumentException("split allows at most one -1 size");
                }
                inferIndex = i;
                continue;
            }
            if (s < 1) {
                throw new IllegalArgumentException("split sizes must be >= 1 (or -1), got " + s);
            }
            knownSum = Math.addExact(knownSum, s);
        }

        long[] resolved = sizes.clone();
        if (inferIndex >= 0) {
            long inferred = axisSize - knownSum;
            if (inferred < 1) {
                throw new IllegalArgumentException(
                        "cannot infer split size: inferred size must be >= 1, got " + inferred);
            }
            resolved[inferIndex] = inferred;
            return resolved;
        }

        if (knownSum != axisSize) {
            throw new IllegalArgumentException(
                    "split sizes must sum to axis size " + axisSize + ", got " + knownSum);
        }
        return resolved;
    }

    private static Tensor concatPair(int axis, Tensor left, Tensor right) {
        if (left.dataType() != right.dataType()) {
            throw new IllegalArgumentException(
                    "concat requires matching dtypes, got "
                            + left.dataType()
                            + " and "
                            + right.dataType());
        }
        if (left.device() != right.device()) {
            throw new IllegalArgumentException(
                    "concat requires matching devices, got "
                            + left.device()
                            + " and "
                            + right.device());
        }

        Shape leftShape = left.shape();
        Shape rightShape = right.shape();
        if (leftShape.rank() != rightShape.rank()) {
            throw new IllegalArgumentException(
                    "concat requires equal ranks, got " + leftShape + " and " + rightShape);
        }

        long[] leftDims = leftShape.toArray();
        long[] rightDims = rightShape.toArray();
        for (int i = 0; i < leftDims.length; i++) {
            if (i == axis) {
                continue;
            }
            if (leftDims[i] != rightDims[i]) {
                throw new IllegalArgumentException(
                        "concat dimension mismatch at axis "
                                + i
                                + ": "
                                + leftDims[i]
                                + " vs "
                                + rightDims[i]);
            }
        }

        long leftAxis = leftDims[axis];
        long rightAxis = rightDims[axis];
        long outAxis = Math.addExact(leftAxis, rightAxis);

        long[] outDims = leftDims.clone();
        outDims[axis] = outAxis;
        Shape outShape = Shape.flat(outDims);

        Tensor idx = Tensor.iota(outAxis, DataType.I32);
        Tensor leftCount = Tensor.scalar(leftAxis, DataType.I32);
        Tensor zeroI32 = Tensor.scalar(0L, DataType.I32);
        Tensor leftMask = idx.lessThan(leftCount);

        Tensor leftIdx = leftMask.where(idx, zeroI32);
        Tensor rightIdx = leftMask.where(zeroI32, idx.subtract(leftCount));

        Tensor leftPart = left.gather(leftIdx, axis);
        Tensor rightPart = right.gather(rightIdx, axis);

        long[] maskDims = new long[leftDims.length];
        Arrays.fill(maskDims, 1L);
        maskDims[axis] = outAxis;
        Tensor axisCoord =
                Tensor.iota(outAxis, DataType.I32).view(Shape.flat(maskDims)).broadcast(outShape);
        Tensor mask = axisCoord.lessThan(leftCount);
        return mask.where(leftPart, rightPart);
    }
}
