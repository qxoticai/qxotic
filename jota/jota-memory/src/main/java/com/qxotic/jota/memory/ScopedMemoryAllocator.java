package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.util.Objects;

/** An allocator whose buffers are {@link ScopedMemory}: each can be freed on its own. */
public interface ScopedMemoryAllocator<B> extends MemoryAllocator<B> {

    ScopedMemory<B> allocateMemory(long byteSize, long byteAlignment);

    default ScopedMemory<B> allocateMemory(long byteSize) {
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default ScopedMemory<B> allocateMemory(
            DataType dataType, long elementCount, long byteAlignment) {
        Objects.requireNonNull(dataType, "dataType");
        if (!supportsDataType(dataType)) {
            throw new IllegalArgumentException("unsupported data type: " + dataType);
        }
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, byteAlignment);
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, long elementCount) {
        return allocateMemory(dataType, elementCount, defaultByteAlignment());
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, Shape shape, long byteAlignment) {
        return allocateMemory(dataType, shape.size(), byteAlignment);
    }

    default ScopedMemory<B> allocateMemory(DataType dataType, Shape shape) {
        return allocateMemory(dataType, shape, defaultByteAlignment());
    }
}
