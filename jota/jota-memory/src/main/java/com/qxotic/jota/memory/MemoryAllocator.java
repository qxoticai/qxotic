package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import java.util.Objects;

public interface MemoryAllocator<B> {
    Device device();

    Memory<B> allocateMemory(long byteSize, long byteAlignment);

    default long defaultByteAlignment() {
        return 1;
    }

    /**
     * Returns the memory allocation granularity in bytes. This is the size of each element in the
     * backing buffer.
     *
     * @return the allocation granularity in bytes
     */
    long memoryGranularity();

    /**
     * Checks whether this allocator can allocate memory for the given data type. Byte-addressable
     * memory supports every data type. Other memory supports data types matching its element size.
     *
     * @param dataType the data type to check
     * @return true if this allocator can allocate the given data type
     */
    default boolean supportsDataType(DataType dataType) {
        long granularity = memoryGranularity();
        return granularity == 1 || dataType.byteSize() == granularity;
    }

    default Memory<B> allocateMemory(long byteSize) {
        return allocateMemory(byteSize, defaultByteAlignment());
    }

    default Memory<B> allocateMemory(DataType dataType, long elementCount, long byteAlignment) {
        Objects.requireNonNull(dataType, "dataType");
        if (!supportsDataType(dataType)) {
            throw new IllegalArgumentException("unsupported data type: " + dataType);
        }
        long byteSize = dataType.byteSizeFor(elementCount);
        return allocateMemory(byteSize, byteAlignment);
    }

    default Memory<B> allocateMemory(DataType dataType, long elementCount) {
        return allocateMemory(dataType, elementCount, defaultByteAlignment());
    }

    default Memory<B> allocateMemory(DataType dataType, Shape shape, long byteAlignment) {
        return allocateMemory(dataType, shape.size(), byteAlignment);
    }

    default Memory<B> allocateMemory(DataType dataType, Shape shape) {
        return allocateMemory(dataType, shape, defaultByteAlignment());
    }
}
