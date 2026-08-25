package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.Stride;
import com.qxotic.jota.ViewTransforms;
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
        return transformed(
                ViewTransforms.reshape(layout, newShape)
                        .orElseThrow(
                                () ->
                                        new IllegalArgumentException(
                                                "cannot reshape view without copying")));
    }

    @Override
    public MemoryView<T> viewCuTe(Shape newShape) {
        var result = ViewTransforms.reshape(layout, newShape);
        if (result.isPresent()) {
            return transformed(result.get());
        }
        if (layout.isSpanContiguous() && layout.isNonOverlapping()) {
            return create(Layout.rowMajor(newShape), dataType, spanStartByteOffset(), memory);
        }
        throw new IllegalArgumentException(
                "cannot reshape non-span-contiguous view without copying");
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
                ViewTransforms.slice(layout, axis, fromInclusive, toExclusive, indexStride));
    }

    private MemoryView<T> transformed(ViewTransforms.Result result) {
        long byteOffsetDelta = Math.multiplyExact(result.elementOffsetDelta(), dataType.byteSize());
        return create(
                result.layout(), dataType, Math.addExact(byteOffset, byteOffsetDelta), memory);
    }

    private long spanStartByteOffset() {
        long elementOffset = 0;
        for (int axis = 0; axis < layout.shape().flatRank(); axis++) {
            long stride = layout.stride().flatAt(axis);
            if (stride < 0) {
                long span = Math.multiplyExact(layout.shape().flatAt(axis) - 1, stride);
                elementOffset = Math.addExact(elementOffset, span);
            }
        }
        return Math.addExact(byteOffset, Math.multiplyExact(elementOffset, dataType.byteSize()));
    }
}
