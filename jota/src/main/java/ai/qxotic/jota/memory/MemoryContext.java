package ai.qxotic.jota.memory;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Device;

public interface MemoryContext<B> extends AutoCloseable {
    Device device();

    MemoryAllocator<B> memoryAllocator();

    /** Optional capability, can be null for opaque memory implementations e.g. GPUs. */
    MemoryAccess<B> memoryAccess();

    MemoryOperations<B> memoryOperations();

    FloatOperations<B> floatOperations();

    /**
     * Returns the memory allocation granularity in bytes. Delegates to the underlying memory
     * allocator.
     *
     * @return the allocation granularity in bytes
     * @see MemoryAllocator#memoryGranularity()
     */
    default long memoryGranularity() {
        return memoryAllocator().memoryGranularity();
    }

    /**
     * Checks if this context can allocate memory for the given DataType. Delegates to the
     * underlying memory allocator.
     *
     * @param dataType the data type to check
     * @return true if this context can allocate the given DataType
     * @see MemoryAllocator#supportsDataType(DataType)
     */
    default boolean supportsDataType(DataType dataType) {
        return memoryAllocator().supportsDataType(dataType);
    }

    @Override
    void close();

    String toString();
}
