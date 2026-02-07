package ai.qxotic.jota.impl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.ir.tir.ViewKind;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class ViewTransforms {

    private ViewTransforms() {}

    /**
     * Result of a view transformation.
     *
     * @param kind the transformation type with parameters (for lazy index computation)
     * @param layout result shape + strides (strides are valid for simple cases, placeholder for
     *     complex cases requiring lazy index computation)
     * @param byteOffsetDelta byte offset adjustment (for slicing)
     * @param needsLazyIndexing true if strides couldn't be computed and lazy index composition is
     *     required
     */
    public record ViewTransformSpec(
            ViewKind kind, Layout layout, long byteOffsetDelta, boolean needsLazyIndexing) {

        /** Convenience constructor for simple cases (no lazy indexing needed). */
        public static ViewTransformSpec simple(ViewKind kind, Layout layout, long byteOffsetDelta) {
            return new ViewTransformSpec(kind, layout, byteOffsetDelta, false);
        }

        /** Convenience constructor for complex cases requiring lazy index computation. */
        public static ViewTransformSpec lazy(ViewKind kind, Layout layout, long byteOffsetDelta) {
            return new ViewTransformSpec(kind, layout, byteOffsetDelta, true);
        }
    }

    public static ViewTransformSpec view(Layout layout, Shape newShape) {
        if (layout.shape().size() != newShape.size()) {
            throw new IllegalArgumentException("total element count mismatch");
        }

        ViewKind kind = new ViewKind.Reshape(layout.shape(), newShape);

        // Fast path: data is contiguous in row-major order
        // Use isSuffixContiguous(0) to verify TRUE row-major contiguity,
        // not just spanning a contiguous range (which includes column-major layouts)
        // Scalars (rank 0) are always trivially contiguous
        if (layout.shape().rank() == 0 || layout.isSuffixContiguous(0)) {
            return ViewTransformSpec.simple(kind, Layout.rowMajor(newShape), 0L);
        }

        long[] oldStrides = layout.stride().toArray();

        // Check if we can compute simple strides
        if (canReshapeWithoutCopy(layout.shape(), newShape, oldStrides)) {
            long[] newStrides = computeReshapeStrides(layout.shape(), newShape, oldStrides);
            Layout newLayout = Layout.of(newShape, Stride.template(newShape, newStrides));
            return ViewTransformSpec.simple(kind, newLayout, 0L);
        }

        // Complex case: strides can't be computed simply, need lazy index composition.
        // Return placeholder row-major strides; the actual indexing will be computed
        // at LIR lowering time by walking the view chain.
        Layout placeholderLayout = Layout.rowMajor(newShape);
        return ViewTransformSpec.lazy(kind, placeholderLayout, 0L);
    }

    public static ViewTransformSpec expand(Layout layout, Shape newShape) {
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

        ViewKind kind = new ViewKind.Expand(currentShape, newShape);
        Stride newStride = Stride.template(newShape, newStrides);
        Layout newLayout = Layout.of(newShape, newStride);
        return ViewTransformSpec.simple(kind, newLayout, 0L);
    }

    public static ViewTransformSpec broadcast(Layout layout, Shape targetShape) {
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

        ViewKind kind = new ViewKind.Broadcast(currentShape, targetShape);

        if (numNewModes == 0) {
            // Same rank, just expand
            ViewTransformSpec expandSpec = expand(layout, targetShape);
            return new ViewTransformSpec(
                    kind,
                    expandSpec.layout(),
                    expandSpec.byteOffsetDelta(),
                    expandSpec.needsLazyIndexing());
        }

        if (!currentShape.isFlat() || !targetShape.isFlat()) {
            long[] newDims = new long[targetShape.flatRank()];
            long[] currentDims = currentShape.toArray();

            int prepend = targetShape.flatRank() - currentShape.flatRank();
            Arrays.fill(newDims, 0, prepend, 1);
            System.arraycopy(currentDims, 0, newDims, prepend, currentDims.length);

            Shape reshapedShape = Shape.flat(newDims);
            ViewTransformSpec reshaped = view(layout, reshapedShape);
            ViewTransformSpec expandSpec = expand(reshaped.layout(), targetShape);
            boolean needsLazy = reshaped.needsLazyIndexing() || expandSpec.needsLazyIndexing();
            return new ViewTransformSpec(kind, expandSpec.layout(), 0L, needsLazy);
        }

        long[] newDims = new long[targetShape.rank()];
        for (int i = 0; i < numNewModes; i++) {
            newDims[i] = 1;
        }
        System.arraycopy(currentShape.toArray(), 0, newDims, numNewModes, currentShape.rank());

        ViewTransformSpec reshaped = view(layout, Shape.flat(newDims));
        ViewTransformSpec expandSpec = expand(reshaped.layout(), targetShape);
        boolean needsLazy = reshaped.needsLazyIndexing() || expandSpec.needsLazyIndexing();
        return new ViewTransformSpec(kind, expandSpec.layout(), 0L, needsLazy);
    }

    public static ViewTransformSpec permute(Layout layout, int... permutationIndices) {
        Shape newShape = layout.shape().permute(permutationIndices);
        Stride newStride = layout.stride().permute(permutationIndices);
        Layout newLayout = Layout.of(newShape, newStride);
        ViewKind kind = new ViewKind.Transpose(permutationIndices.clone());
        return ViewTransformSpec.simple(kind, newLayout, 0L);
    }

    public static ViewTransformSpec transpose(Layout layout, int _axis0, int _axis1) {
        Shape shape = layout.shape();
        int axis0 = Util.wrapAround(_axis0, shape.rank());
        int axis1 = Util.wrapAround(_axis1, shape.rank());
        int[] permutation = new int[shape.rank()];
        for (int i = 0; i < shape.rank(); i++) {
            permutation[i] = i;
        }
        permutation[axis0] = axis1;
        permutation[axis1] = axis0;
        return permute(layout, permutation);
    }

    public static ViewTransformSpec slice(
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
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
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
                                "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
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

        ViewKind kind = new ViewKind.Slice(axis, fromInclusive, indexStride);
        Layout newLayout = Layout.of(newShape, newStride);
        return ViewTransformSpec.simple(kind, newLayout, byteOffsetDelta);
    }

    /**
     * Returns true if the layout spans a contiguous memory range.
     *
     * <p>Delegates to {@link Layout#spansContiguousRange()} for CuTe-style contiguity checking. A
     * layout spans a contiguous range if all elements fit within a contiguous block of memory
     * without gaps: sum((dim_i - 1) * stride_i) == totalElements - 1.
     */
    private static boolean spansContiguousRange(Layout layout) {
        return layout.spansContiguousRange();
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
