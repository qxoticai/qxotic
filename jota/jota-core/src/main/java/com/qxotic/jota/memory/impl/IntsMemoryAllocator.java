package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class IntsMemoryAllocator implements MemoryAllocator<int[]> {

    private static final IntsMemoryAllocator INSTANCE = new IntsMemoryAllocator();

    public static MemoryAllocator<int[]> instance() {
        return INSTANCE;
    }

    private IntsMemoryAllocator() {}

    @Override
    public Device device() {
        return new Device(DeviceType.JAVA, 0);
    }

    @Override
    public long defaultByteAlignment() {
        return Integer.BYTES;
    }

    @Override
    public long memoryGranularity() {
        return Integer.BYTES;
    }

    @Override
    public Memory<int[]> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        if (byteSize % Integer.BYTES != 0) {
            throw new IllegalArgumentException("unaligned (4) byteSize");
        }
        int length = Math.toIntExact(byteSize / Integer.BYTES);
        return MemoryFactory.ofInts(new int[length]);
    }
}
