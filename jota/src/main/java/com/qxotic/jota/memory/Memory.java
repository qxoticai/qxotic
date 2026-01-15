package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;

public interface Memory<B> {

    long byteSize();

    boolean isReadOnly();

    Device device();

    B base();

    /**
     * Returns the memory access granularity in bytes.
     * This is the size of each element in the backing buffer.
     *
     * @return the access granularity in bytes
     */
    long memoryGranularity();

    /**
     * Checks if this memory can store the given DataType.
     * A DataType is supported if its byteSize is a multiple of the memory granularity.
     *
     * @param dataType the data type to check
     * @return true if this memory can store the given DataType
     */
    default boolean supportsDataType(DataType dataType) {
        long granularity = memoryGranularity();
        return granularity == 1 || dataType.byteSize() == granularity;
    }
}
