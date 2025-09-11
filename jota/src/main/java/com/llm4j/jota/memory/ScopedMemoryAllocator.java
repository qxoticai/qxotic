package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Shape;

public interface ScopedMemoryAllocator<B> extends MemoryAllocator<B> {

    ScopedMemory<B> allocateMemory(long byteSize, long byteAlignment);

    default ScopedMemory<B> allocateMemory(long byteSize) {
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, long elementCount, long byteAlignment) {
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, byteAlignment);
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, long elementCount) {
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, Shape shape, long byteAlignment) {
        return allocateMemory(dataType, shape.totalNumberOfElements(), byteAlignment);
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, Shape shape) {
        return allocateMemory(dataType, shape, defaultByteAlignment());
    }
}
