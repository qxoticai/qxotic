package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Stride;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryView;

import java.util.ArrayList;
import java.util.List;

final class MemoryViewImpl<T> implements MemoryView<T> {

    private final Layout layout;
    private final DataType dataType;
    private final Memory<T> memory;
    private final long byteOffset;
    final boolean isContiguous;

    private MemoryViewImpl(Layout layout, DataType dataType, long byteOffset, Memory<T> memory) {
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        this.layout = layout;
        this.dataType = dataType;
        this.memory = memory;
        this.byteOffset = byteOffset;
        this.isContiguous = layout.isCongruentWith(Layout.rowMajor(layout.shape()));

        if (this.isContiguous) {
            if (byteOffset + dataType.byteSizeFor(layout.shape().size()) > memory.byteSize()) {
                throw new IllegalArgumentException("view spans beyond memory size");
            }
        } else {
            // TODO(peterssen): Check view is within memory bounds, taking strides into account.
        }
    }

    // Legacy constructor for backward compatibility
    private MemoryViewImpl(Shape shape, long[] byteStrides, DataType dataType, long byteOffset, Memory<T> memory) {
        // Convert byte strides to element strides
        long elementByteSize = dataType.byteSize();
        long[] elementStrides = new long[byteStrides.length];
        for (int i = 0; i < elementStrides.length; i++) {
            elementStrides[i] = byteStrides[i] / elementByteSize;
        }
        this.layout = Layout.of(shape, Stride.flat(elementStrides));
        this.dataType = dataType;
        this.byteOffset = byteOffset;
        this.memory = memory;
        this.isContiguous = MemoryView.super.isContiguous();

        if (this.isContiguous) {
            if (byteOffset + dataType.byteSizeFor(shape.size()) > memory.byteSize()) {
                throw new IllegalArgumentException("view spans beyond memory size");
            }
        } else {
            // TODO(peterssen): Check view is within memory bounds, taking strides into account.
        }
    }

    static <B> MemoryView<B> create(Shape shape, long[] byteStrides, DataType dataType, long byteOffset, Memory<B> memory) {
        return new MemoryViewImpl<>(shape, byteStrides, dataType, byteOffset, memory);
    }

    static <B> MemoryView<B> create(Layout layout, DataType dataType, long byteOffset, Memory<B> memory) {
        return new MemoryViewImpl<>(layout, dataType, byteOffset, memory);
    }

    @Override
    public Layout layout() {
        return layout;
    }

    @Override
    public long byteOffset() {
        return byteOffset;
    }

    @Override
    public Memory<T> memory() {
        return memory;
    }

    @Override
    public DataType dataType() {
        return dataType;
    }

    @Override
    public boolean isContiguous() {
        return isContiguous;
    }

    @Override
    public MemoryView<T> reshape(Shape newShape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MemoryView<T> permute(int... permutationIndices) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MemoryView<T> expand(Shape newShape) {
        throw new UnsupportedOperationException();
    }

    @Override
    public MemoryView<T> slice(int _axis, long fromInclusive, long toExclusive, long indexStride) {
        throw new UnsupportedOperationException();
    }

//    @Override
//    public MemoryView<T> permute(int... permutationIndices) {
//        Shape newShape = shape.permute(permutationIndices);
//
//        long[] newStrides = new long[newShape.rank()];
//        for (int i = 0; i < permutationIndices.length; i++) {
//            int srcIndex = permutationIndices[i];
//            newStrides[i] = byteStrides[srcIndex];
//        }
//
//        return MemoryViewImpl.create(newShape, newStrides, dataType(), byteOffset(), memory());
//    }
//
//    @Override
//    public MemoryView<T> expand(Shape newShape) {
//        if (newShape == null || newShape.rank() == 0) {
//            throw new IllegalArgumentException("New shape must be non-null and non-empty");
//        }
//
//        int oldRank = shape().rank();
//        int newRank = newShape.rank();
//        if (oldRank != newRank) {
//            throw new IllegalArgumentException(
//                    "Expand must preserve rank: expected " + oldRank + " dimensions, got " + newRank
//            );
//        }
//
//        // Validate dims: allow zero-sized dims
//        for (int i = 0; i < newRank; i++) {
//            long oldDim = shape().dimension(i);
//            long newDim = newShape.dimension(i);
//
//            if (newDim < 0) {
//                throw new IllegalArgumentException("Invalid dimension in shape: " + newDim);
//            }
//
//            // If newDim == 0, always allow (result is empty)
//            if (newDim == 0) {
//                continue;
//            }
//
//            // Otherwise, standard broadcasting rule per axis:
//            // either we broadcast from 1 -> newDim, or we must match exactly.
//            if (oldDim != 1 && newDim != oldDim) {
//                throw new IllegalArgumentException(
//                        "Dimension " + i + ": cannot expand size " + oldDim + " into " + newDim
//                );
//            }
//        }
//
//        // Compute new strides
//        long[] oldStrides = byteStrides();
//        long[] newStrides = new long[newRank];
//        for (int i = 0; i < newRank; i++) {
//            long oldDim = shape().dimension(i);
//            long newDim = newShape.dimension(i);
//
//            if (newDim == 0) {
//                // Zero-sized axis: keep original stride unless it was broadcasted;
//                // no elements are accessed anyway, but this avoids flagging as broadcast.
//                newStrides[i] = (oldDim == 1) ? 0 : oldStrides[i];
//            } else if (oldDim == 1) {
//                // Broadcasted dimension
//                newStrides[i] = 0;
//            } else {
//                // Preserved dimension
//                newStrides[i] = oldStrides[i];
//            }
//        }
//
//        return new MemoryViewImpl<>(
//                newShape,
//                newStrides,
//                dataType(),
//                byteOffset(),
//                memory()
//        );
//    }

//
//    @Override
//    public MemoryView<T> slice(int _axis, long fromInclusive, long toExclusive) {
//        int axis = shape.wrapAround(_axis);
//        long dimSize = shape.dimensionAt(axis);
//        if (fromInclusive < 0 || toExclusive > dimSize || fromInclusive > toExclusive) {
//            throw new IllegalArgumentException(String.format(
//                    "Invalid slice range [%d, %d) for dimension %d of size %d",
//                    fromInclusive, toExclusive, axis, dimSize
//            ));
//        }
//
//        // Calculate new offset
//        long newOffset = byteOffset + fromInclusive * byteStrides[axis];
//
//        // Calculate new shape
//        long[] newDims = shape.toArray().clone();
//        newDims[axis] = toExclusive - fromInclusive;
//        Shape newShape = Shape.of(newDims);
//
//        // Copy existing strides
//        long[] newStrides = byteStrides.clone();
//
//        return new MemoryViewImpl<>(
//                newShape,
//                newStrides,
//                dataType,
//                newOffset,
//                memory
//        );
//    }
//
//    @Override
//    public MemoryView<T> slice(int _axis, long fromInclusive, long toExclusive, long indexStride) {
//        int axis = shape.wrapAround(_axis);
//        long dimSize = shape.dimension(axis);
//
//        // Validate step
//        if (indexStride == 0) {
//            throw new IllegalArgumentException("Step cannot be zero");
//        }
//
//        // Validate slice bounds (adjust for negative step)
//        if (indexStride > 0) {
//            if (fromInclusive < 0 || toExclusive > dimSize || fromInclusive > toExclusive) {
//                throw new IllegalArgumentException(String.format(
//                        "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
//                        fromInclusive, toExclusive, indexStride, axis, dimSize
//                ));
//            }
//        } else { // step < 0
//            if (fromInclusive >= dimSize || toExclusive < -1 || fromInclusive < toExclusive) {
//                throw new IllegalArgumentException(String.format(
//                        "Invalid slice range [%d, %d) with step %d for dimension %d of size %d",
//                        fromInclusive, toExclusive, indexStride, axis, dimSize
//                ));
//            }
//        }
//
//        // Calculate new offset (accounting for step direction)
//        long newOffset = byteOffset + fromInclusive * byteStrides[axis];
//
//        // Calculate new shape
//        long[] newDims = shape.toArray().clone();
//        long length;
//        if (indexStride > 0) {
//            length = (toExclusive - fromInclusive + indexStride - 1) / indexStride; // ceil division
//        } else {
//            length = (toExclusive - fromInclusive + indexStride + 1) / indexStride; // ceil division for negative step
//        }
//        newDims[axis] = length > 0 ? length : 0;
//        Shape newShape = Shape.of(newDims);
//
//        // Copy and adjust strides
//        long[] newStrides = byteStrides.clone();
//        newStrides[axis] = byteStrides[axis] * indexStride;
//
//        return new MemoryViewImpl<>(
//                newShape,
//                newStrides,
//                dataType,
//                newOffset,
//                memory
//        );
//    }
//
//
//    @Override
//    public MemoryView<T> reshape(Shape newShape) {
//        if (shape().size() != newShape.size()) {
//            throw new IllegalArgumentException("total element count mismatch");
//        }
//
//        // If contiguous, we can always reshape by recomputing strides
//        if (isContiguous()) {
//            long[] newStrides = computeContiguousStrides(newShape, dataType());
//            return new MemoryViewImpl<>(newShape, newStrides, dataType(), byteOffset(), memory());
//        }
//
//        // For non-contiguous views, we need to check if reshape is possible without copy
//        if (!canReshapeWithoutCopy(shape(), newShape, byteStrides(), dataType())) {
//            throw new IllegalArgumentException("Cannot reshape: would require copying data");
//        }
//
//        // Compute new strides for the reshaped view
//        long[] newStrides = computeReshapeStrides(shape(), newShape, byteStrides(), dataType());
//
//        return new MemoryViewImpl<>(newShape, newStrides, dataType(), byteOffset(), memory());
//    }
//
//    /**
//     * Computes contiguous strides for a given shape.
//     */
//    private static long[] computeContiguousStrides(Shape shape, DataType dataType) {
//        long[] strides = new long[shape.rank()];
//        if (shape.rank() == 0) {
//            return strides;
//        }
//
//        long stride = dataType.byteSize();
//        for (int i = shape.rank() - 1; i >= 0; i--) {
//            strides[i] = stride;
//            stride *= shape.dimension(i);
//        }
//        return strides;
//    }
//
//    /**
//     * Checks if we can reshape from oldShape to newShape without copying data.
//     */
//    private static boolean canReshapeWithoutCopy(Shape oldShape, Shape newShape, long[] oldStrides, DataType dataType) {
//        // Create lists of non-singleton dimensions and their strides
//        java.util.List<Long> oldDims = new ArrayList<>();
//        List<Long> oldStridesFiltered = new ArrayList<>();
//
//        for (int i = 0; i < oldShape.rank(); i++) {
//            long dim = oldShape.dimension(i);
//            if (dim != 1) {
//                oldDims.add(dim);
//                oldStridesFiltered.add(oldStrides[i]);
//            }
//        }
//
//        List<Long> newDims = new ArrayList<>();
//        for (int i = 0; i < newShape.rank(); i++) {
//            long dim = newShape.dimension(i);
//            if (dim != 1) {
//                newDims.add(dim);
//            }
//        }
//
//        // If both have same non-singleton dimensions, it's just adding/removing 1s
//        if (oldDims.equals(newDims)) {
//            return true;
//        }
//
//        // Check if we can group consecutive dimensions
//        return canGroupDimensions(oldDims, newDims, oldStridesFiltered, dataType);
//    }
//
//    /**
//     * Checks if oldDims can be regrouped into newDims by collapsing/expanding contiguous dimensions.
//     */
//    private static boolean canGroupDimensions(List<Long> oldDims, List<Long> newDims,
//                                              List<Long> oldStrides, DataType dataType) {
//        if (oldDims.isEmpty() && newDims.isEmpty()) {
//            return true;
//        }
//
//        int oldIdx = 0, newIdx = 0;
//
//        while (oldIdx < oldDims.size() && newIdx < newDims.size()) {
//            long oldProduct = 1;
//            long newProduct = 1;
//            int oldStart = oldIdx;
//
//            // Accumulate old dimensions until we have enough elements
//            while (oldIdx < oldDims.size() && oldProduct < newDims.get(newIdx)) {
//                oldProduct *= oldDims.get(oldIdx);
//                oldIdx++;
//            }
//
//            // Accumulate new dimensions until we match the old product
//            while (newIdx < newDims.size() && newProduct < oldProduct) {
//                newProduct *= newDims.get(newIdx);
//                newIdx++;
//            }
//
//            // Products must match exactly
//            if (oldProduct != newProduct) {
//                return false;
//            }
//
//            // Check if the grouped old dimensions are contiguous in memory
//            if (!areContiguous(oldDims, oldStrides, oldStart, oldIdx, dataType)) {
//                return false;
//            }
//        }
//
//        return oldIdx == oldDims.size() && newIdx == newDims.size();
//    }
//
//    /**
//     * Checks if dimensions from startIdx to endIdx (exclusive) are contiguous in memory.
//     */
//    private static boolean areContiguous(List<Long> dims, List<Long> strides,
//                                         int startIdx, int endIdx, DataType dataType) {
//        if (endIdx - startIdx <= 1) {
//            return true; // Single dimension is always contiguous
//        }
//
//        // Check if strides follow the pattern for contiguous memory (from right to left)
//        long expectedStride = dataType.byteSize();
//
//        for (int i = endIdx - 1; i >= startIdx; i--) {
//            if (strides.get(i) != expectedStride) {
//                return false;
//            }
//            expectedStride *= dims.get(i);
//        }
//
//        return true;
//    }
//
//    private static long[] computeReshapeStrides(Shape oldShape, Shape newShape, long[] oldStrides, DataType dataType) {
//        List<Long> oldDimsNonSingleton = new ArrayList<>();
//        List<Long> oldStridesNonSingleton = new ArrayList<>();
//        for (int i = 0; i < oldShape.rank(); i++) {
//            if (oldShape.dimension(i) != 1) {
//                oldDimsNonSingleton.add(oldShape.dimension(i));
//                oldStridesNonSingleton.add(oldStrides[i]);
//            }
//        }
//
//        List<Long> newDimsNonSingleton = new ArrayList<>();
//        List<Integer> newDimIndices = new ArrayList<>();
//        for (int i = 0; i < newShape.rank(); i++) {
//            if (newShape.dimension(i) != 1) {
//                newDimsNonSingleton.add(newShape.dimension(i));
//                newDimIndices.add(i);
//            }
//        }
//
//        long[] newStrides = new long[newShape.rank()];
//
//        // Fast path: only adding/removing singleton dimensions
//        if (oldDimsNonSingleton.equals(newDimsNonSingleton)) {
//            // Copy strides of non-singleton axes in order
//            int k = 0;
//            for (int i = 0; i < newShape.rank(); i++) {
//                if (newShape.dimension(i) != 1) {
//                    newStrides[i] = oldStridesNonSingleton.get(k++);
//                }
//            }
//            // Fill singleton axes with a non-zero stride (copy the stride to the right; last axis uses element size)
//            for (int i = newShape.rank() - 1; i >= 0; i--) {
//                if (newShape.dimension(i) == 1) {
//                    newStrides[i] = (i == newShape.rank() - 1)
//                            ? dataType.byteSize()
//                            : newStrides[i + 1];
//                }
//            }
//            return newStrides;
//        }
//
//        // General regrouping path (when products are regrouped).
//        // Existing logic, but DO NOT prefill singletons with zero.
//        int oldIdx = 0, newIdx = 0;
//        while (oldIdx < oldDimsNonSingleton.size() && newIdx < newDimsNonSingleton.size()) {
//            long oldProduct = 1, newProduct = 1;
//            int oldStart = oldIdx;
//            int newStart = newIdx;
//
//            while (oldIdx < oldDimsNonSingleton.size() && oldProduct < newDimsNonSingleton.get(newIdx)) {
//                oldProduct *= oldDimsNonSingleton.get(oldIdx++);
//            }
//            while (newIdx < newDimsNonSingleton.size() && newProduct < oldProduct) {
//                newProduct *= newDimsNonSingleton.get(newIdx++);
//            }
//            if (oldProduct != newProduct) {
//                throw new IllegalArgumentException("Cannot reshape: incompatible dimension grouping");
//            }
//
//            long baseStride = oldStridesNonSingleton.get(oldStart);
//
//            // Assign strides for non-singleton axes in this group (rightmost gets baseStride)
//            for (int i = newStart; i < newIdx; i++) {
//                int actualNewIdx = newDimIndices.get(i);
//                if (i == newIdx - 1) {
//                    newStrides[actualNewIdx] = baseStride;
//                } else {
//                    long s = baseStride;
//                    for (int j = i + 1; j < newIdx; j++) {
//                        int laterIdx = newDimIndices.get(j);
//                        s *= newShape.dimension(laterIdx);
//                    }
//                    newStrides[actualNewIdx] = s;
//                }
//            }
//        }
//
//        // Fill remaining singleton axes with a non-zero stride (copy right neighbor or element size)
//        for (int i = newShape.rank() - 1; i >= 0; i--) {
//            if (newStrides[i] == 0) {
//                newStrides[i] = (i == newShape.rank() - 1)
//                        ? dataType.byteSize()
//                        : newStrides[i + 1];
//            }
//        }
//
//        return newStrides;
//    }
}
