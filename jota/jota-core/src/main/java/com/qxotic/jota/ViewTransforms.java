package com.qxotic.jota;

import java.util.Objects;
import java.util.Optional;

/** Exact affine transformations of a layout. */
public final class ViewTransforms {

    private ViewTransforms() {}

    /** A transformed layout and its origin relative to the input layout, in elements. */
    public record Result(Layout layout, long elementOffsetDelta) {
        public Result {
            Objects.requireNonNull(layout);
        }
    }

    /** Returns an exact view reshape, or empty when reshaping requires reindexing or copying. */
    public static Optional<Result> reshape(Layout layout, Shape shape) {
        if (layout.shape().size() != shape.size()) {
            throw new IllegalArgumentException("total element count mismatch");
        }
        if (layout.shape().equals(shape)) {
            return Optional.of(result(layout));
        }
        if (shape.size() == 0 || layout.shape().isScalar()) {
            return Optional.of(result(Layout.rowMajor(shape)));
        }

        long[] strides = reshapeStrides(layout, shape);
        if (strides == null) {
            return Optional.empty();
        }
        return Optional.of(result(Layout.of(shape, Stride.template(shape, strides))));
    }

    /** Inserts a zero-stride singleton mode. */
    public static Result unsqueeze(Layout layout, int axis) {
        int normalizedAxis = Util.wrapAround(axis, layout.shape().rank() + 1);
        return result(
                Layout.of(
                        layout.shape().insert(normalizedAxis, Shape.of(1)),
                        layout.stride().insert(normalizedAxis, Stride.of(0))));
    }

    /** Expands singleton dimensions without changing the shape structure. */
    public static Result expand(Layout layout, Shape shape) {
        Shape currentShape = layout.shape();
        if (!currentShape.isCongruentWith(shape)) {
            throw new IllegalArgumentException(
                    "expand requires congruent shapes: current="
                            + currentShape
                            + ", target="
                            + shape);
        }

        long[] strides = layout.stride().toArray();
        for (int i = 0; i < currentShape.flatRank(); i++) {
            long current = currentShape.flatAt(i);
            long target = shape.flatAt(i);
            if (current == target) {
                continue;
            }
            if (current != 1) {
                throw new IllegalArgumentException(
                        "cannot expand dimension " + i + " from " + current + " to " + target);
            }
            strides[i] = 0;
        }
        return result(Layout.of(shape, Stride.template(shape, strides)));
    }

    /** Broadcasts a layout by aligning its flattened dimensions from the right. */
    public static Result broadcast(Layout layout, Shape shape) {
        Shape currentShape = layout.shape();
        int leadingDimensions = shape.flatRank() - currentShape.flatRank();
        if (leadingDimensions < 0) {
            throw new IllegalArgumentException(
                    "cannot broadcast shape " + currentShape + " to shape " + shape);
        }

        long[] currentStrides = layout.stride().toArray();
        long[] strides = new long[shape.flatRank()];
        for (int i = 0; i < currentShape.flatRank(); i++) {
            long current = currentShape.flatAt(i);
            long target = shape.flatAt(leadingDimensions + i);
            if (current == target) {
                strides[leadingDimensions + i] = currentStrides[i];
            } else if (current != 1) {
                throw new IllegalArgumentException(
                        "cannot broadcast dimension " + i + " from " + current + " to " + target);
            }
        }
        return result(Layout.of(shape, Stride.template(shape, strides)));
    }

    /** Permutes top-level layout modes. */
    public static Result permute(Layout layout, int... permutation) {
        for (int axis : permutation) {
            if (axis < 0) {
                throw new IllegalArgumentException("negative axis in permutation: " + axis);
            }
        }
        return result(
                Layout.of(
                        layout.shape().permute(permutation), layout.stride().permute(permutation)));
    }

    /** Selects a strided range from one top-level mode. */
    public static Result slice(
            Layout layout, int axis, long fromInclusive, long toExclusive, long step) {
        int normalizedAxis = Util.wrapAround(axis, layout.shape().rank());
        long dimension = layout.shape().size(normalizedAxis);
        validateSlice(dimension, normalizedAxis, fromInclusive, toExclusive, step);

        Layout mode = layout.modeAt(normalizedAxis).coalesce();
        if (mode.shape().rank() != 1) {
            throw new IllegalArgumentException(
                    "cannot slice non-affine nested mode " + normalizedAxis + ": " + mode);
        }

        long modeStride = mode.stride().flatAt(0);
        long length = sliceLength(fromInclusive, toExclusive, step);
        Shape shape = layout.shape().replace(normalizedAxis, Shape.flat(length));
        Stride stride =
                layout.stride()
                        .replace(normalizedAxis, Stride.of(Math.multiplyExact(modeStride, step)));
        return new Result(Layout.of(shape, stride), Math.multiplyExact(fromInclusive, modeStride));
    }

    private static Result result(Layout layout) {
        return new Result(layout, 0);
    }

    private static void validateSlice(
            long dimension, int axis, long fromInclusive, long toExclusive, long step) {
        if (step == 0) {
            throw new IllegalArgumentException("step cannot be zero");
        }
        boolean invalid =
                step > 0
                        ? fromInclusive < 0
                                || toExclusive > dimension
                                || fromInclusive > toExclusive
                        : fromInclusive < 0
                                || fromInclusive >= dimension
                                || toExclusive < -1
                                || toExclusive >= dimension
                                || fromInclusive < toExclusive;
        if (invalid) {
            throw new IllegalArgumentException(
                    "invalid slice range ["
                            + fromInclusive
                            + ", "
                            + toExclusive
                            + ") with step "
                            + step
                            + " for dimension "
                            + axis
                            + " of size "
                            + dimension);
        }
    }

    private static long sliceLength(long fromInclusive, long toExclusive, long step) {
        long distance = step > 0 ? toExclusive - fromInclusive : fromInclusive - toExclusive;
        if (distance == 0) {
            return 0;
        }
        if (step == Long.MIN_VALUE) {
            return 1;
        }
        long magnitude = Math.abs(step);
        return 1 + (distance - 1) / magnitude;
    }

    /** Returns new flattened strides, or {@code null} when no affine reshape exists. */
    private static long[] reshapeStrides(Layout layout, Shape shape) {
        Shape currentShape = layout.shape();
        int currentRank = currentShape.flatRank();
        if (currentRank == 0) {
            return Stride.rowMajor(shape).toArray();
        }

        long[] currentStrides = layout.stride().toArray();
        long[] strides = new long[shape.flatRank()];
        int targetAxis = shape.flatRank() - 1;
        long chunkStride = currentStrides[currentRank - 1];
        long currentElements = 1;
        long targetElements = 1;

        for (int currentAxis = currentRank - 1; currentAxis >= 0; currentAxis--) {
            currentElements = Math.multiplyExact(currentElements, currentShape.flatAt(currentAxis));
            boolean chunkBoundary =
                    currentAxis == 0
                            || currentShape.flatAt(currentAxis - 1) != 1
                                    && currentStrides[currentAxis - 1]
                                            != Math.multiplyExact(currentElements, chunkStride);
            if (!chunkBoundary) {
                continue;
            }

            while (targetAxis >= 0
                    && (targetElements < currentElements || shape.flatAt(targetAxis) == 1)) {
                strides[targetAxis] = Math.multiplyExact(targetElements, chunkStride);
                targetElements = Math.multiplyExact(targetElements, shape.flatAt(targetAxis));
                targetAxis--;
            }
            if (targetElements != currentElements) {
                return null;
            }

            if (currentAxis > 0) {
                chunkStride = currentStrides[currentAxis - 1];
                currentElements = 1;
                targetElements = 1;
            }
        }
        return targetAxis == -1 ? strides : null;
    }
}
