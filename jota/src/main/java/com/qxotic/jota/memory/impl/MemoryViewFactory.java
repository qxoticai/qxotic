package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Layout;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryView;

public final class MemoryViewFactory {

    private MemoryViewFactory() {
        // no instances
    }

    public static <B> MemoryView<B> of(DataType dataType, Memory<B> memory, long byteOffset, Layout layout) {
        return MemoryViewImpl.create(layout, dataType, byteOffset, memory);
    }

    public static <B> MemoryView<B> of(DataType dataType, Memory<B> memory, Layout layout) {
        return MemoryViewImpl.create(layout, dataType, 0L, memory);
    }

    public static <B> MemoryView<B> rowMajor(DataType dataType, Memory<B> memory, Shape shape) {
        return of(dataType, memory, 0L, Layout.rowMajor(shape));
    }

    public static <B> MemoryView<B> allocate(MemoryAllocator<B> memoryAllocator, DataType dataType, Shape shape) {
        // TODO: rowMajor by default?
        return rowMajor(dataType, memoryAllocator.allocateMemory(dataType.byteSizeFor(shape)), shape);
    }
}
