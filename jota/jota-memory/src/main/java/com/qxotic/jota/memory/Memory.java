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

    /** The backing storage (an array, {@code ByteBuffer}, {@code MemorySegment}, ...). */
    B base();

    /**
     * The access granularity in bytes: 1 for byte-addressable memory, else the size of one backing
     * element.
     */
    long memoryGranularity();

    /**
     * Byte-addressable memory supports every data type; other memory supports data types whose
     * {@link DataType#byteSize()} matches {@link #memoryGranularity()}.
     */
    default boolean supportsDataType(DataType dataType) {
        long granularity = memoryGranularity();
        return granularity == 1 || dataType.byteSize() == granularity;
    }
}
