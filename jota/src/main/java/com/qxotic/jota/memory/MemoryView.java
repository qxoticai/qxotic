package com.qxotic.jota.memory;

import com.qxotic.jota.memory.impl.MemoryViewFactory;
import com.qxotic.jota.*;

import java.util.Arrays;
import java.util.stream.IntStream;

public interface MemoryView<B> {

    Layout layout();

    default Shape shape() {
        return layout().shape();
    }

    default Stride stride() {
        return layout().stride();
    }

    DataType dataType();

    default Stride byteStride() {
        return stride().scale(dataType().byteSize());
    }

    Memory<B> memory();

    long byteOffset();

    default boolean isBroadcasted() {
        return Arrays.stream(stride().toArray()).anyMatch(stride -> stride == 0L);
    }

    default boolean isContiguous() {
        if (shape().hasZeroElements()) {
            return true; // Empty views are trivially contiguous
        }
        long expectedStride = 1; // element stride for contiguous layout
        long[] strides = stride().toArray();
        for (int i = shape().flatRank() - 1; i >= 0; i--) {
            if (strides[i] != expectedStride) {
                return false;
            }
            expectedStride *= shape().flatAt(i);
        }
        return true;
    }

    static boolean isWithinBounds(Layout layout, DataType dataType, Memory<?> memory, long byteOffset) {
        if (layout.shape().size() == 0) {
            return true;
        }

        long minRelativeOffset = 0;
            long maxRelativeOffset = 0;
        long[] strides = layout.stride().scale(dataType.byteSize()).toArray();
        for (int i = 0; i < layout.shape().flatRank(); i++) {
            long dim = layout.shape().flatAt(i);
            if (dim <= 1) {
                continue;
            }
            long stride = strides[i];
            long span = (dim - 1) * stride;
            if (stride >= 0) {
                maxRelativeOffset += span;
            } else {
                minRelativeOffset += span;
            }
        }
        long minOffset = byteOffset + minRelativeOffset;
        long maxOffset = byteOffset + maxRelativeOffset;
        return minOffset >= 0 && maxOffset + dataType.byteSize() <= memory.byteSize();
    }

    MemoryView<B> view(Shape newShape);

    MemoryView<B> permute(int... permutationIndices);

    MemoryView<B> expand(Shape newShape);

    MemoryView<B> slice(int _axis, long fromInclusive, long toExclusive, long indexStride);

    default MemoryView<B> slice(int _axis, long fromInclusive, long toExclusive) {
        return slice(_axis, fromInclusive, toExclusive, 1);
    }

    default MemoryView<B> transpose(int _axis0, int _axis1) {
        Shape shape = shape();
        int axis0 = Util.wrapAround(_axis0, shape.rank());
        int axis1 = Util.wrapAround(_axis1, shape.rank());
        int[] permutation = IntStream.range(0, shape.rank()).toArray();
        permutation[axis0] = axis1;
        permutation[axis1] = axis0;
        return permute(permutation);
    }

    default MemoryView<B> broadcast(Shape targetShape) {
        // Step 1: Reshape by adding singleton dimensions if needed (without copying)
        Shape currentShape = shape();

        // Determine how many dimensions we need to add
        int numNewDims = targetShape.rank() - currentShape.rank();
        if (numNewDims < 0) {
            throw new IllegalArgumentException("Cannot broadcast shape " + currentShape
                    + " to shape " + targetShape + ": has fewer dimensions");
        }

        // Prepend singleton dimensions
        long[] newDims = new long[targetShape.rank()];
        for (int i = 0; i < numNewDims; i++) {
            newDims[i] = 1;
        }
        System.arraycopy(
                currentShape.toArray(), 0,
                newDims, numNewDims,
                currentShape.rank()
        );

        // Create the reshaped view
        MemoryView<B> reshaped = this.view(Shape.flat(newDims));

        // Step 2: Expand singleton dimensions to target sizes
        return reshaped.expand(targetShape);
    }

    // Factory methods
    static <B> MemoryView<B> of(Memory<B> memory, long byteOffset, DataType dtype, Layout layout) {
        return MemoryViewFactory.of(dtype, memory, byteOffset, layout);
    }

    static <B> MemoryView<B> of(Memory<B> memory, DataType dtype, Layout layout) {
        return MemoryViewFactory.of(dtype, memory, 0, layout);
    }

    static <B> MemoryView<B> rowMajor(Memory<B> memory, DataType dtype, Shape shape) {
        return of(memory, 0, dtype, Layout.rowMajor(shape));
    }

    default MemoryView<B> withLayout(Layout newLayout) {
        return of(memory(), byteOffset(), dataType(), newLayout);
    }

    default MemoryView<B> withStride(Stride newStride) {
        return withLayout(Layout.of(shape(), newStride));
    }

    default MemoryView<B> broadcastTo(Shape targetShape) {
        // Delegate to existing broadcast method for now
        return broadcast(targetShape);
    }

    default String toString(MemoryAccess<B> memoryAccess) {
        return MemoryViewPrinter.toString(this, memoryAccess);
    }

    default String toString(MemoryAccess<B> memoryAccess, ViewPrintOptions options) {
        return MemoryViewPrinter.toString(this, memoryAccess, options);
    }
}

