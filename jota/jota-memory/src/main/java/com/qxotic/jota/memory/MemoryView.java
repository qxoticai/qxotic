package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.Util;
import com.qxotic.jota.View;
import com.qxotic.jota.ViewTransforms;
import com.qxotic.jota.memory.impl.MemoryViewFactory;
import java.util.Arrays;
import java.util.stream.IntStream;

public interface MemoryView<B> extends View {

    Layout layout();

    default Shape shape() {
        return layout().shape();
    }

    default Stride stride() {
        return layout().stride();
    }

    DataType dataType();

    Memory<B> memory();

    long byteOffset();

    default Stride byteStride() {
        return stride().scale(dataType().byteSize());
    }

    default boolean isBroadcasted() {
        return Arrays.stream(stride().toArray()).anyMatch(stride -> stride == 0L);
    }

    default boolean isContiguous() {
        return layout().isSpanContiguous();
    }

    default boolean isSpanContiguous() {
        return layout().isSpanContiguous();
    }

    default boolean isRowMajorContiguous() {
        return layout().isRowMajorContiguous();
    }

    default boolean isNonOverlapping() {
        return layout().isNonOverlapping();
    }

    static boolean isWithinBounds(
            Layout layout, DataType dataType, Memory<?> memory, long byteOffset) {
        if (layout.shape().size() == 0) {
            return true;
        }
        // Exact math: a span that overflows long is out of bounds by definition.
        try {
            long minRelativeOffset = 0;
            long maxRelativeOffset = 0;
            long[] strides = layout.stride().scale(dataType.byteSize()).toArray();
            for (int i = 0; i < layout.shape().flatRank(); i++) {
                long dim = layout.shape().flatAt(i);
                if (dim <= 1) {
                    continue;
                }
                long span = Math.multiplyExact(dim - 1, strides[i]);
                if (span >= 0) {
                    maxRelativeOffset = Math.addExact(maxRelativeOffset, span);
                } else {
                    minRelativeOffset = Math.addExact(minRelativeOffset, span);
                }
            }
            long minOffset = Math.addExact(byteOffset, minRelativeOffset);
            long maxOffset = Math.addExact(byteOffset, maxRelativeOffset);
            return minOffset >= 0
                    && Math.addExact(maxOffset, dataType.byteSize()) <= memory.byteSize();
        } catch (ArithmeticException overflow) {
            return false;
        }
    }

    MemoryView<B> view(Shape newShape);

    /**
     * Reshapes a view using CuTe span-contiguous semantics.
     *
     * <p>This allows reshaping any span-contiguous, non-overlapping view even if the linear order
     * changes. The result uses row-major strides for the new shape.
     */
    MemoryView<B> viewCuTe(Shape newShape);

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
        ViewTransforms.Result result = ViewTransforms.broadcast(layout(), targetShape);
        return of(memory(), byteOffset(), dataType(), result.layout());
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

    default String toString(MemoryAccess<B> memoryAccess) {
        return MemoryViewPrinter.toString(this, memoryAccess);
    }

    default String toString(MemoryAccess<B> memoryAccess, ViewPrintOptions options) {
        return MemoryViewPrinter.toString(this, memoryAccess, options);
    }
}
