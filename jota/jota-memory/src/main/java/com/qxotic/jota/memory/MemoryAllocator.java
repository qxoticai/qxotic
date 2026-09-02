package com.qxotic.jota.memory;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.Shape;
import java.util.Objects;

/** Allocates {@link Memory} of a requested byte size and alignment on a device. */
public interface MemoryAllocator<B> {
    Device device();

    Memory<B> allocateMemory(long byteSize, long byteAlignment);

    /** The alignment used by the {@code allocateMemory} overloads without one. */
    default long defaultByteAlignment() {
        return 1;
    }

    /**
     * The allocation granularity in bytes: 1 for byte-addressable memory, else the size of one
     * backing element.
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
