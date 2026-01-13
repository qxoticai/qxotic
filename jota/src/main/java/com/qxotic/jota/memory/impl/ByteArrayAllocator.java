package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class ByteArrayAllocator implements MemoryAllocator<byte[]> {

    private static final ByteArrayAllocator INSTANCE = new ByteArrayAllocator();

    public static MemoryAllocator<byte[]> instance() {
        return INSTANCE;
    }

    private ByteArrayAllocator() {
    }

    private static final byte[] EMPTY = new byte[0];

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public Memory<byte[]> allocateMemory(long byteSize, long byteAlignment) {
        if (byteAlignment != 1) {
            throw new IllegalArgumentException("byteAlignment != 1");
        }
        int length = Math.toIntExact(byteSize);
        return ByteArrayMemory.of(new byte[length]);
    }
}
