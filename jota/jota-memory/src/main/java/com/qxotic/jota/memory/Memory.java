package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;

/**
 * A block of bytes on some device, backed by {@code B} (an array, a buffer, a segment, a handle).
 */
public interface Memory<B> {

    long byteSize();

    boolean isReadOnly();

    Device device();

    B base();

    /**
     * Returns the memory access granularity in bytes. This is the size of each element in the
     * backing buffer.
     *
     * @return the access granularity in bytes
     */
    long memoryGranularity();

    /**
     * Checks whether this memory can store the given data type. Byte-addressable memory supports
     * every data type. Other memory supports data types matching its element size.
     *
     * @param dataType the data type to check
     * @return true if this memory can store the given data type
     */
    default boolean supportsDataType(DataType dataType) {
        long granularity = memoryGranularity();
        return granularity == 1 || dataType.byteSize() == granularity;
    }
}
