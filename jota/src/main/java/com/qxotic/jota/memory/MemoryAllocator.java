package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;

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
        return allocateMemory(dataType, shape.size(), byteAlignment);
    }

    default Memory<B> allocateMemory(DataType dataType, Shape shape) {
        return allocateMemory(dataType, shape, defaultByteAlignment());
    }
}

