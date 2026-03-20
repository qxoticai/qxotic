package com.qxotic.jota.memory.impl;

import com.qxotic.jota.DataType;
import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class BooleansMemoryAllocator implements MemoryAllocator<boolean[]> {

    private static final BooleansMemoryAllocator INSTANCE = new BooleansMemoryAllocator();

    public static MemoryAllocator<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansMemoryAllocator() {}

    @Override
    public Device device() {
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public boolean supportsDataType(DataType dataType) {
        return dataType == DataType.BOOL; // ONLY BOOL - override default behavior
    }

    @Override
    public Memory<boolean[]> allocateMemory(long byteSize, long byteAlignment) {
        if (byteAlignment != 1) {
            throw new IllegalArgumentException("byteAlignment != 1");
        }
        int length = Math.toIntExact(byteSize); // 1 byte per boolean
        return BooleansMemory.of(new boolean[length]);
    }
}
