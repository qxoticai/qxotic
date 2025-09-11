package com.llm4j.jota.memory.impl;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;
import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryAllocator;
import com.llm4j.jota.memory.MemoryView;

public final class MemoryViewFactory {

    private MemoryViewFactory() {
        // no instances
    }

    public static <B> MemoryView<B> of(Shape shape, long[] byteStrides, DataType dataType, long byteOffset, Memory<B> memory) {
        return MemoryViewImpl.create(shape, byteStrides, dataType, byteOffset, memory);
    }

    public static <B> MemoryView<B> of(Shape shape, DataType dataType, long byteOffset, Memory<B> memory) {
        return of(shape, MemoryViewImpl.contiguousByteStrides(shape, dataType), dataType, byteOffset, memory);
    }

    public static <B> MemoryView<B> of(Shape shape, DataType dataType, Memory<B> memory) {
        return of(shape, dataType, 0L, memory);
    }

    public static <B> MemoryView<B> allocate(Shape shape, DataType dataType, MemoryAllocator<B> memoryAllocator) {
        return of(shape, dataType, memoryAllocator.allocateMemory(dataType.byteSizeFor(shape)));
    }
}
