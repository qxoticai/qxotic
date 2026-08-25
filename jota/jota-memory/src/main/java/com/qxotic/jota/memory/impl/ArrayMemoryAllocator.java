package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import java.util.Objects;
import java.util.function.IntFunction;

final class ArrayMemoryAllocator<B> implements MemoryAllocator<B> {

    private static final Device DEVICE = DeviceType.JAVA.deviceIndex(0);

    private final Class<B> arrayType;
    private final int elementByteSize;
    private final IntFunction<B> arrayFactory;

    ArrayMemoryAllocator(Class<B> arrayType, IntFunction<B> arrayFactory) {
        this.arrayType = Objects.requireNonNull(arrayType);
        this.elementByteSize = ArrayMemory.elementByteSize(arrayType);
        this.arrayFactory = Objects.requireNonNull(arrayFactory);
    }

    @Override
    public Device device() {
        return DEVICE;
    }

    @Override
    public long defaultByteAlignment() {
        return elementByteSize;
    }

    @Override
    public long memoryGranularity() {
        return elementByteSize;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return arrayType == boolean[].class
                ? dataType == DataType.BOOL
                : MemoryAllocator.super.supportsDataType(dataType);
    }

    @Override
    public Memory<B> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("negative byte size");
        }
        if (byteAlignment <= 0 || elementByteSize % byteAlignment != 0) {
            throw new IllegalArgumentException("unsupported byte alignment: " + byteAlignment);
        }
        if (byteSize % elementByteSize != 0) {
            throw new IllegalArgumentException(
                    "byte size is not a multiple of " + elementByteSize + ": " + byteSize);
        }

        int length = Math.toIntExact(byteSize / elementByteSize);
        return ArrayMemory.of(arrayType.cast(arrayFactory.apply(length)));
    }
}
