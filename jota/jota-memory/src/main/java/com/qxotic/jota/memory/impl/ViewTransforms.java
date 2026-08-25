package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.Util;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class ViewTransforms {

    private ViewTransforms() {}

    /**
     * Result of a view transformation.
     *
     * @param layout result shape + strides (strides are valid for simple cases, placeholder for
     *     complex cases requiring lazy index computation)
     * @param byteOffsetDelta byte offset adjustment (for slicing)
     * @param needsLazyIndexing whether consumers must preserve the logical index mapping
     */
    public record Result(Layout layout, long byteOffsetDelta, boolean needsLazyIndexing) {}

    private static Result simple(Layout layout, long byteOffsetDelta) {
        return new Result(layout, byteOffsetDelta, false);
    }

    private static Result lazy(Layout layout, long byteOffsetDelta) {
        return new Result(layout, byteOffsetDelta, true);
    }

    public static Result view(Layout layout, Shape newShape) {
        if (layout.shape().size() != newShape.size()) {
            throw new IllegalArgumentException("total element count mismatch");
        }

        // Fast path: data is contiguous in row-major order
        // Use isSuffixContiguous(0) to verify TRUE row-major contiguity,
        // not just spanning a contiguous range (which includes column-major layouts)
        // Scalars (rank 0) are always trivially contiguous
        if (layout.shape().rank() == 0 || layout.isSuffixContiguous(0)) {
            return simple(Layout.rowMajor(newShape), 0L);
        }

        long[] oldStrides = layout.stride().toArray();

        // Check if we can compute simple strides
        if (canReshapeWithoutCopy(layout.shape(), newShape, oldStrides)) {
            long[] newStrides = computeReshapeStrides(layout.shape(), newShape, oldStrides);
            Layout newLayout = Layout.of(newShape, Stride.template(newShape, newStrides));
            return simple(newLayout, 0L);
        }

        // Complex case: strides alone cannot represent the reshape.
        // Preserve the output shape and signal that index mapping is required.
        Layout placeholderLayout = Layout.rowMajor(newShape);
        return lazy(placeholderLayout, 0L);
    }

    /**
     * Inserts a singleton mode with zero stride.
     *
     * <p>This preserves data aliasing semantics (PyTorch-style unsqueeze): the inserted axis has
     * shape=1 and stride=0.
     */
    public static Result unsqueeze(Layout layout, int axis_) {
        int axis = Util.wrapAround(axis_, layout.shape().rank() + 1);
        Shape newShape = layout.shape().insert(axis, Shape.of(1));
        Stride newStride = layout.stride().insert(axis, Stride.of(0));
        Layout newLayout = Layout.of(newShape, newStride);
        return simple(newLayout, 0L);
    }

    public static Result expand(Layout layout, Shape newShape) {
        Shape currentShape = layout.shape();

        if (!currentShape.isCongruentWith(newShape)) {
            throw new IllegalArgumentException(
                    "expand requires congruent shapes: current="
                            + currentShape
                            + ", target="
                            + newShape);
        }

        long[] currentStrides = layout.stride().toArray();
        long[] newStrides = new long[currentShape.flatRank()];

        for (int i = 0; i < currentShape.flatRank(); i++) {
            long currentDim = currentShape.flatAt(i);
            long newDim = newShape.flatAt(i);

            if (currentDim == 1) {
                newStrides[i] = 0;
            } else if (currentDim == newDim) {
                newStrides[i] = currentStrides[i];
            } else {
                throw new IllegalArgumentException(
                        "Cannot expand dimension "
                                + i
                                + " from size "
                                + currentDim
                                + " to "
                                + newDim);
            }
        }

        Stride newStride = Stride.template(newShape, newStrides);
        Layout newLayout = Layout.of(newShape, newStride);
        return simple(newLayout, 0L);
    }

    public static Result broadcast(Layout layout, Shape targetShape) {
        Shape currentShape = layout.shape();
        int numNewModes = targetShape.rank() - currentShape.rank();
        if (numNewModes < 0) {
            throw new IllegalArgumentException(
                    "Cannot broadcast shape "
                            + currentShape
                            + " to shape "
                            + targetShape
                            + ": target has fewer modes");
        }

        if (numNewModes == 0) {
            // Same rank, just expand
            return expand(layout, targetShape);
        }

        if (!currentShape.isFlat() || !targetShape.isFlat()) {
            long[] newDims = new long[targetShape.flatRank()];
            long[] currentDims = currentShape.toArray();

            int prepend = targetShape.flatRank() - currentShape.flatRank();
            Arrays.fill(newDims, 0, prepend, 1);
            System.arraycopy(currentDims, 0, newDims, prepend, currentDims.length);

            Shape reshapedShape = Shape.flat(newDims);
            Result reshaped = view(layout, reshapedShape);
            Result expanded = expand(reshaped.layout(), targetShape);
            boolean needsLazy = reshaped.needsLazyIndexing() || expanded.needsLazyIndexing();
            return new Result(expanded.layout(), 0L, needsLazy);
        }

        long[] newDims = new long[targetShape.rank()];
        for (int i = 0; i < numNewModes; i++) {
            newDims[i] = 1;
        }
        System.arraycopy(currentShape.toArray(), 0, newDims, numNewModes, currentShape.rank());

        Result reshaped = view(layout, Shape.flat(newDims));
        Result expanded = expand(reshaped.layout(), targetShape);
        boolean needsLazy = reshaped.needsLazyIndexing() || expanded.needsLazyIndexing();
        return new Result(expanded.layout(), 0L, needsLazy);
    }

    public static Result permute(Layout layout, int... permutationIndices) {
        for (int axis : permutationIndices) {
            if (axis < 0) {
                throw new IllegalArgumentException("negative axis in permutation: " + axis);
            }
        }
        Shape newShape = layout.shape().permute(permutationIndices);
        Stride newStride = layout.stride().permute(permutationIndices);
        Layout newLayout = Layout.of(newShape, newStride);
        return simple(newLayout, 0L);
    }

    public static Result slice(
            Layout layout,
            DataType dataType,
            int _axis,
            long fromInclusive,
            long toExclusive,
            long indexStride) {
        int axis = Util.wrapAround(_axis, layout.shape().rank());
        long dimSize = layout.shape().size(axis);

        if (indexStride == 0) {
            throw new IllegalArgumentException("Step cannot be zero");
        }

        if (indexStride > 0) {
            if (fromInclusive < 0 || toExclusive > dimSize || fromInclusive > toExclusive) {
                throw new IllegalArgumentException(
                        String.format(
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size"
                                        + " %d",
                                fromInclusive, toExclusive, indexStride, axis, dimSize));
            }
        } else {
            if (fromInclusive < 0
                    || fromInclusive >= dimSize
                    || toExclusive < -1
                    || toExclusive >= dimSize
                    || fromInclusive < toExclusive) {
                throw new IllegalArgumentException(
                        String.format(
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size"
                                        + " %d",
                                fromInclusive, toExclusive, indexStride, axis, dimSize));
            }
        }

        long byteStride = layout.stride().modeAt(axis).flatAt(0) * dataType.byteSize();
        long byteOffsetDelta = fromInclusive * byteStride;

        long length;
        if (indexStride > 0) {
            length = (toExclusive - fromInclusive + indexStride - 1) / indexStride;
        } else {
            length = (toExclusive - fromInclusive + indexStride + 1) / indexStride;
        }
        if (length < 0) {
            length = 0;
        }

        Shape newModeShape = Shape.flat(length);
        Shape newShape = layout.shape().replace(axis, newModeShape);
        Stride newModeStride = layout.stride().modeAt(axis).scale(indexStride);
        Stride newStride = layout.stride().replace(axis, newModeStride);

        Layout newLayout = Layout.of(newShape, newStride);
        return lazy(newLayout, byteOffsetDelta);
    }

    private static boolean canReshapeWithoutCopy(
            Shape oldShape, Shape newShape, long[] oldStrides) {
        List<Long> oldDims = new ArrayList<>();
        List<Long> oldStridesFiltered = new ArrayList<>();
        for (int i = 0; i < oldShape.flatRank(); i++) {
            long dim = oldShape.flatAt(i);
            if (dim != 1) {
                oldDims.add(dim);
                oldStridesFiltered.add(oldStrides[i]);
            }
        }

        List<Long> newDims = new ArrayList<>();
        for (int i = 0; i < newShape.flatRank(); i++) {
            long dim = newShape.flatAt(i);
            if (dim != 1) {
                newDims.add(dim);
            }
        }

        if (oldDims.equals(newDims)) {
            // When non-singleton dims match (just adding/removing size-1 dimensions),
            // the existing strides are valid for the new shape. This handles:
            // - Squeeze: (2,3,1):(15,5,1) -> (2,3):(15,5)
            // - Unsqueeze: (2,3):(15,5) -> (2,3,1):(15,5,?)
            // - Same-shape no-op: (4,3):(1,4) -> (4,3):(1,4)
            return true;
        }

        return canGroupDimensions(oldDims, newDims, oldStridesFiltered);
    }

    private static boolean canGroupDimensions(
            List<Long> oldDims, List<Long> newDims, List<Long> oldStrides) {
        if (oldDims.isEmpty() && newDims.isEmpty()) {
            return true;
        }

        int oldIdx = 0;
        int newIdx = 0;
        while (oldIdx < oldDims.size() && newIdx < newDims.size()) {
            long oldProduct = 1;
            long newProduct = 1;
            int oldStart = oldIdx;

            while (oldIdx < oldDims.size() && oldProduct < newDims.get(newIdx)) {
                oldProduct *= oldDims.get(oldIdx);
                oldIdx++;
            }

            while (newIdx < newDims.size() && newProduct < oldProduct) {
                newProduct *= newDims.get(newIdx);
                newIdx++;
            }

            if (oldProduct != newProduct) {
                return false;
            }

            if (!areContiguous(oldDims, oldStrides, oldStart, oldIdx)) {
                return false;
            }
        }

        return oldIdx == oldDims.size() && newIdx == newDims.size();
    }

    /**
     * Checks if strides within the given range form a row-major pattern.
     *
     * <p>For reshape without copy to work, the strides must satisfy:
     *
     * <ol>
     *   <li>The innermost stride must be 1
     *   <li>stride[i] == stride[i+1] * dim[i+1] for all i in [startIdx, endIdx-1)
     * </ol>
     *
     * <p>This ensures that iterating in row-major order visits elements in the same order as their
     * memory layout.
     */
    private static boolean areContiguous(
            List<Long> dims, List<Long> strides, int startIdx, int endIdx) {
        if (endIdx <= startIdx) {
            return true;
        }

        // The innermost stride must be 1 for true contiguity
        if (strides.get(endIdx - 1) != 1) {
            return false;
        }

        // Verify row-major ordering: stride[i] == stride[i+1] * dim[i+1]
        for (int i = startIdx; i < endIdx - 1; i++) {
            long expectedStride = strides.get(i + 1) * dims.get(i + 1);
            if (strides.get(i) != expectedStride) {
                return false;
            }
        }

        return true;
    }

    private static long[] computeReshapeStrides(Shape oldShape, Shape newShape, long[] oldStrides) {
        List<Long> oldDimsNonSingleton = new ArrayList<>();
        List<Long> oldStridesNonSingleton = new ArrayList<>();
        for (int i = 0; i < oldShape.flatRank(); i++) {
            if (oldShape.flatAt(i) != 1) {
                oldDimsNonSingleton.add(oldShape.flatAt(i));
                oldStridesNonSingleton.add(oldStrides[i]);
            }
        }

        List<Long> newDimsNonSingleton = new ArrayList<>();
        List<Integer> newDimIndices = new ArrayList<>();
        for (int i = 0; i < newShape.flatRank(); i++) {
            if (newShape.flatAt(i) != 1) {
                newDimsNonSingleton.add(newShape.flatAt(i));
                newDimIndices.add(i);
            }
        }

        long[] newStrides = new long[newShape.flatRank()];

        if (oldDimsNonSingleton.equals(newDimsNonSingleton)) {
            int k = 0;
            for (int i = 0; i < newShape.flatRank(); i++) {
                if (newShape.flatAt(i) != 1) {
                    newStrides[i] = oldStridesNonSingleton.get(k++);
                }
            }
            for (int i = newShape.flatRank() - 1; i >= 0; i--) {
                if (newShape.flatAt(i) == 1) {
                    newStrides[i] = (i == newShape.flatRank() - 1) ? 1 : newStrides[i + 1];
                }
            }
            return newStrides;
        }

        int oldIdx = 0;
        int newIdx = 0;
        while (oldIdx < oldDimsNonSingleton.size() && newIdx < newDimsNonSingleton.size()) {
            long oldProduct = 1;
            long newProduct = 1;
            int oldStart = oldIdx;
            int newStart = newIdx;

            while (oldIdx < oldDimsNonSingleton.size()
                    && oldProduct < newDimsNonSingleton.get(newIdx)) {
                oldProduct *= oldDimsNonSingleton.get(oldIdx++);
            }
            while (newIdx < newDimsNonSingleton.size() && newProduct < oldProduct) {
                newProduct *= newDimsNonSingleton.get(newIdx++);
            }
            if (oldProduct != newProduct) {
                throw new IllegalArgumentException(
                        "Cannot reshape: incompatible dimension grouping");
            }

            long baseStride = oldStridesNonSingleton.get(oldStart);
            for (int i = newStart; i < newIdx; i++) {
                int actualNewIdx = newDimIndices.get(i);
                if (i == newIdx - 1) {
                    newStrides[actualNewIdx] = baseStride;
                } else {
                    long stride = baseStride;
                    for (int j = i + 1; j < newIdx; j++) {
                        int laterIdx = newDimIndices.get(j);
                        stride *= newShape.flatAt(laterIdx);
                    }
                    newStrides[actualNewIdx] = stride;
                }
            }
        }

        for (int i = newShape.flatRank() - 1; i >= 0; i--) {
            if (newStrides[i] == 0) {
                newStrides[i] = (i == newShape.flatRank() - 1) ? 1 : newStrides[i + 1];
            }
        }

        return newStrides;
    }
}
