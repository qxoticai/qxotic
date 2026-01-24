package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Layout;
import ai.qxotic.jota.Shape;
import ai.qxotic.jota.Stride;
import ai.qxotic.jota.Util;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import ai.qxotic.jota.memory.MemoryViewPrinter;
import ai.qxotic.jota.memory.ViewPrintOptions;
import java.util.ArrayList;
import java.util.List;

final class MemoryViewImpl<T> implements MemoryView<T> {

    private final Layout layout;
    private final DataType dataType;
    private final Stride byteStride;
    private final Memory<T> memory;
    private final long byteOffset;
    final boolean isContiguous;

    @Override
    public Stride byteStride() {
        return this.byteStride;
    }

    private MemoryViewImpl(Layout layout, DataType dataType, long byteOffset, Memory<T> memory) {
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        this.layout = layout;
        this.dataType = dataType;
        this.byteStride = this.layout.stride().scale(this.dataType.byteSize());
        this.memory = memory;
        this.byteOffset = byteOffset;
        this.isContiguous = isContiguous(layout);

        if (!memory.supportsDataType(dataType)) {
            throw new IllegalArgumentException("unsupported data type: " + dataType);
        }
        if (!MemoryView.isWithinBounds(layout, dataType, memory, byteOffset)) {
            throw new IllegalArgumentException("view spans beyond memory size");
        }
    }

    static <B> MemoryView<B> create(
            Layout layout, DataType dataType, long byteOffset, Memory<B> memory) {
        return new MemoryViewImpl<>(layout, dataType, byteOffset, memory);
    }

    private static boolean isContiguous(Layout layout) {
        if (layout.shape().hasZeroElements()) {
            return true;
        }
        long expectedStride = 1;
        long[] strides = layout.stride().toArray();
        for (int i = layout.shape().flatRank() - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= layout.shape().flatAt(i);
        }
        return true;
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
    public String toString() {
        return MemoryViewPrinter.toString(this);
    }

    @Override
    public String toString(MemoryAccess<T> memoryAccess) {
        return MemoryViewPrinter.toString(this, memoryAccess);
    }

    @Override
    public String toString(MemoryAccess<T> memoryAccess, ViewPrintOptions options) {
        return MemoryViewPrinter.toString(this, memoryAccess, options);
    }

    @Override
    public MemoryView<T> view(Shape newShape) {
        if (layout.shape().size() != newShape.size()) {
            throw new IllegalArgumentException("total element count mismatch");
        }

        // CuTe semantics: if the layout spans a contiguous memory range [0, n-1],
        // the new shape gets row-major strides (linear memory iteration order).
        if (spansContiguousRange()) {
            return MemoryViewImpl.create(Layout.rowMajor(newShape), dataType, byteOffset, memory);
        }

        // For non-contiguous layouts, check if reshape is still possible
        long[] oldStrides = layout.stride().toArray();
        if (!canReshapeWithoutCopy(layout.shape(), newShape, oldStrides)) {
            throw new IllegalArgumentException("Cannot view: would require copying data");
        }

        long[] newStrides = computeReshapeStrides(layout.shape(), newShape, oldStrides);
        Layout newLayout = Layout.of(newShape, Stride.template(newShape, newStrides));
        return MemoryViewImpl.create(newLayout, dataType, byteOffset, memory);
    }

    /**
     * Checks if this layout spans a contiguous memory range [0, n-1].
     *
     * <p>This is true if: sum((dim_i - 1) * stride_i) == totalElements - 1
     *
     * <p>This is more general than row-major contiguity. For example, (2, 2, 2):(4, 1, 2) spans [0,
     * 7] contiguously even though iteration order is not linear.
     */
    private boolean spansContiguousRange() {
        if (layout.shape().hasZeroElements()) {
            return true;
        }
        long span = 0;
        long totalElements = 1;
        long[] strides = layout.stride().toArray();
        for (int i = 0; i < layout.shape().flatRank(); i++) {
            long dim = layout.shape().flatAt(i);
            span += (dim - 1) * strides[i];
            totalElements *= dim;
        }
        return span == totalElements - 1;
    }

    @Override
    public MemoryView<T> permute(int... permutationIndices) {
        Shape newShape = layout.shape().permute(permutationIndices);
        Stride newStride = layout.stride().permute(permutationIndices);
        Layout newLayout = Layout.of(newShape, newStride);
        return MemoryViewImpl.create(newLayout, dataType, byteOffset, memory);
    }

    @Override
    public MemoryView<T> expand(Shape newShape) {
        Shape currentShape = layout.shape();

        // Must have same rank (same nesting structure)
        if (!currentShape.isCongruentWith(newShape)) {
            throw new IllegalArgumentException(
                    "expand requires congruent shapes: current="
                            + currentShape
                            + ", target="
                            + newShape);
        }

        // Validate expansion is possible and compute new strides
        long[] currentStrides = layout.stride().toArray();
        long[] newStrides = new long[currentShape.flatRank()];

        for (int i = 0; i < currentShape.flatRank(); i++) {
            long currentDim = currentShape.flatAt(i);
            long newDim = newShape.flatAt(i);

            if (currentDim == 1) {
                // Broadcasting: stride becomes 0
                newStrides[i] = 0;
            } else if (currentDim == newDim) {
                // Dimension matches: preserve stride
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

        // Create new layout preserving nesting structure
        Stride newStride = Stride.template(newShape, newStrides);
        Layout newLayout = Layout.of(newShape, newStride);

        return MemoryViewImpl.create(newLayout, dataType, byteOffset, memory);
    }

    @Override
    public MemoryView<T> slice(int _axis, long fromInclusive, long toExclusive, long indexStride) {
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

        long newOffset = byteOffset + fromInclusive * byteStride.modeAt(axis).flatAt(0);

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
        return MemoryViewImpl.create(newLayout, dataType, newOffset, memory);
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
     * Checks if a group of dimensions spans a contiguous memory range.
     *
     * <p>For dimensions with sizes (d0, d1, ...) and strides (s0, s1, ...), they span a contiguous
     * range if: sum((di - 1) * si) == product(di) - 1
     *
     * <p>This is more general than row-major contiguity. For example, (2, 2):(1, 2) spans offsets
     * {0, 1, 2, 3} which is contiguous, even though the access order is [0, 2, 1, 3].
     *
     * <p>Note: A single dimension with stride > 1 has holes and is NOT contiguous.
     */
    private static boolean areContiguous(
            List<Long> dims, List<Long> strides, int startIdx, int endIdx) {
        if (endIdx - startIdx == 0) {
            return true; // Empty range is trivially contiguous
        }

        // Calculate span: sum of (dim - 1) * stride for each dimension
        long span = 0;
        long totalElements = 1;
        for (int i = startIdx; i < endIdx; i++) {
            span += (dims.get(i) - 1) * strides.get(i);
            totalElements *= dims.get(i);
        }

        // Dimensions span a contiguous range if max_offset == total_elements - 1
        // (min offset is 0 at all-zero indices, and valid layouts have no duplicates)
        return span == totalElements - 1;
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
    //        return MemoryViewImpl.create(newShape, newStrides, dataType(), byteOffset(),
    // memory());
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
    //                    "Expand must preserve rank: expected " + oldRank + " dimensions, got " +
    // newRank
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
    //                        "Dimension " + i + ": cannot expand size " + oldDim + " into " +
    // newDim
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
    //    public MemoryView<T> slice(int _axis, long fromInclusive, long toExclusive, long
    // indexStride) {
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
    //                        "Invalid slice range [%d, %d) with step %d for dimension %d of size
    // %d",
    //                        fromInclusive, toExclusive, indexStride, axis, dimSize
    //                ));
    //            }
    //        } else { // step < 0
    //            if (fromInclusive >= dimSize || toExclusive < -1 || fromInclusive < toExclusive) {
    //                throw new IllegalArgumentException(String.format(
    //                        "Invalid slice range [%d, %d) with step %d for dimension %d of size
    // %d",
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
    //            length = (toExclusive - fromInclusive + indexStride - 1) / indexStride; // ceil
    // division
    //        } else {
    //            length = (toExclusive - fromInclusive + indexStride + 1) / indexStride; // ceil
    // division for negative step
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
    //            return new MemoryViewImpl<>(newShape, newStrides, dataType(), byteOffset(),
    // memory());
    //        }
    //
    //        // For non-contiguous views, we need to check if reshape is possible without copy
    //        if (!canReshapeWithoutCopy(shape(), newShape, byteStrides(), dataType())) {
    //            throw new IllegalArgumentException("Cannot reshape: would require copying data");
    //        }
    //
    //        // Compute new strides for the reshaped view
    //        long[] newStrides = computeReshapeStrides(shape(), newShape, byteStrides(),
    // dataType());
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
    //    private static boolean canReshapeWithoutCopy(Shape oldShape, Shape newShape, long[]
    // oldStrides, DataType dataType) {
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
    //     * Checks if oldDims can be regrouped into newDims by collapsing/expanding contiguous
    // dimensions.
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
    //    private static long[] computeReshapeStrides(Shape oldShape, Shape newShape, long[]
    // oldStrides, DataType dataType) {
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
    //            // Fill singleton axes with a non-zero stride (copy the stride to the right; last
    // axis uses element size)
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
    //            while (oldIdx < oldDimsNonSingleton.size() && oldProduct <
    // newDimsNonSingleton.get(newIdx)) {
    //                oldProduct *= oldDimsNonSingleton.get(oldIdx++);
    //            }
    //            while (newIdx < newDimsNonSingleton.size() && newProduct < oldProduct) {
    //                newProduct *= newDimsNonSingleton.get(newIdx++);
    //            }
    //            if (oldProduct != newProduct) {
    //                throw new IllegalArgumentException("Cannot reshape: incompatible dimension
    // grouping");
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
    //        // Fill remaining singleton axes with a non-zero stride (copy right neighbor or
    // element size)
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
