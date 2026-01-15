package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;
import ai.qxotic.jota.Shape;

public interface MemoryAllocator<B> {
    Device device();

    Memory<B> allocateMemory(long byteSize, long byteAlignment);

    default long defaultByteAlignment() {
        return 1;
    }

    /**
     * Returns the memory allocation granularity in bytes.
     * This is the size of each element in the backing buffer.
     *
     * @return the allocation granularity in bytes
     */
    long memoryGranularity();

    /**
     * Checks if this allocator can allocate memory for the given DataType.
     * A DataType is supported if its byteSize is a multiple of the memory granularity.
     *
     * @param dataType the data type to check
     * @return true if this allocator can allocate the given DataType
     */
    default boolean supportsDataType(DataType dataType) {
        long granularity = memoryGranularity();
        return granularity == 1 || dataType.byteSize() == granularity;
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

