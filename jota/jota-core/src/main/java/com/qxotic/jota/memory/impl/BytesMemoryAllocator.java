package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class BytesMemoryAllocator implements MemoryAllocator<byte[]> {

    private static final BytesMemoryAllocator INSTANCE = new BytesMemoryAllocator();

    public static MemoryAllocator<byte[]> instance() {
        return INSTANCE;
    }

    private BytesMemoryAllocator() {}

    private static final byte[] EMPTY = new byte[0];

    @Override
    public Device device() {
        return new Device(DeviceType.JAVA, 0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<byte[]> allocateMemory(long byteSize, long byteAlignment) {
        if (byteAlignment != 1) {
            throw new IllegalArgumentException("byteAlignment != 1");
        }
        int length = Math.toIntExact(byteSize);
        return BytesMemory.of(new byte[length]);
    }
}
