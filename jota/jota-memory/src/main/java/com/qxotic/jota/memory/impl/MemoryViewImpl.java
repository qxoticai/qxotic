package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.jota.memory.MemoryViewPrinter;
import com.qxotic.jota.memory.ViewPrintOptions;

final class MemoryViewImpl<T> implements MemoryView<T> {

    private final Layout layout;
    private final DataType dataType;
    private final Stride byteStride;
    private final Memory<T> memory;
    private final long byteOffset;

    private MemoryViewImpl(Layout layout, DataType dataType, long byteOffset, Memory<T> memory) {
        if (byteOffset < 0) {
            throw new IllegalArgumentException("negative offset");
        }
        this.layout = layout;
        this.dataType = dataType;
        this.byteStride = layout.stride().scale(dataType.byteSize());
        this.memory = memory;
        this.byteOffset = byteOffset;

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

    @Override
    public Layout layout() {
        return layout;
    }

    @Override
    public long byteOffset() {
        return byteOffset;
    }

    @Override
    public Stride byteStride() {
        return byteStride;
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
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(layout, newShape);
        if (spec.needsLazyIndexing()) {
            throw new IllegalArgumentException(
                    "Cannot reshape non-contiguous view without copying. "
                            + "Use Tensor.view() which handles this automatically, "
                            + "or make a contiguous copy first.");
        }
        return transformed(spec);
    }

    @Override
    public MemoryView<T> viewCuTe(Shape newShape) {
        ViewTransforms.ViewTransformSpec spec = ViewTransforms.view(layout, newShape);
        if (spec.needsLazyIndexing()) {
            if (layout.isSpanContiguous() && layout.isNonOverlapping()) {
                return create(Layout.rowMajor(newShape), dataType, byteOffset, memory);
            }
            throw new IllegalArgumentException(
                    "Cannot reshape non-span-contiguous view without copying. "
                            + "Use Tensor.view() to preserve linear order, "
                            + "or make a contiguous copy first.");
        }
        return transformed(spec);
    }

    @Override
    public MemoryView<T> permute(int... permutationIndices) {
        return transformed(ViewTransforms.permute(layout, permutationIndices));
    }

    @Override
    public MemoryView<T> expand(Shape newShape) {
        return transformed(ViewTransforms.expand(layout, newShape));
    }

    @Override
    public MemoryView<T> slice(int axis, long fromInclusive, long toExclusive, long indexStride) {
        return transformed(
                ViewTransforms.slice(
                        layout, dataType, axis, fromInclusive, toExclusive, indexStride));
    }

    private MemoryView<T> transformed(ViewTransforms.ViewTransformSpec spec) {
        return create(spec.layout(), dataType, byteOffset + spec.byteOffsetDelta(), memory);
    }
}
