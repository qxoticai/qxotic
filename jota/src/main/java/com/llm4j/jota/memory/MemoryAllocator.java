package com.llm4j.jota.memory;

import com.llm4j.jota.DataType;
import com.llm4j.jota.Device;
import com.llm4j.jota.Shape;

public interface MemoryAllocator<B> {
    Device device();

    Memory<B> allocateMemory(long byteSize, long byteAlignment);

    default long defaultByteAlignment() {
        return 1;
    }

    default Memory<B> allocateMemory(long byteSize) {
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default Memory<B> allocateMemory(DataType dataType, long elementCount, long byteAlignment) {
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, byteAlignment);
    }

    default Memory<B> allocateMemory(DataType dataType, long elementCount) {
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default Memory<B> allocateMemory(DataType dataType, Shape shape, long byteAlignment) {
        return allocateMemory(dataType, shape.totalNumberOfElements(), byteAlignment);
    }

    default Memory<B> allocateMemory(DataType dataType, Shape shape) {
        return allocateMemory(dataType, shape, defaultByteAlignment());
    }
}

