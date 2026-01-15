package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class ShortsMemoryAllocator implements MemoryAllocator<short[]> {

    private static final ShortsMemoryAllocator INSTANCE = new ShortsMemoryAllocator();

    public static MemoryAllocator<short[]> instance() {
        return INSTANCE;
    }

    private ShortsMemoryAllocator() {
    }

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public long defaultByteAlignment() {
        return Short.BYTES;
    }

    @Override
    public long memoryGranularity() {
        return Short.BYTES;
    }

    @Override
    public Memory<short[]> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        if (byteSize % Short.BYTES != 0) {
            throw new IllegalArgumentException("unaligned (2) byteSize");
        }
        int length = Math.toIntExact(byteSize / Short.BYTES);
        return MemoryFactory.ofShorts(new short[length]);
    }
}
