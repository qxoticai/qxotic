package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class BytesMemoryAllocator implements MemoryAllocator<byte[]> {

    private static final BytesMemoryAllocator INSTANCE = new BytesMemoryAllocator();

    public static MemoryAllocator<byte[]> instance() {
        return INSTANCE;
    }

    private BytesMemoryAllocator() {}

    private static final byte[] EMPTY = new byte[0];

    @Override
    public Device device() {
        return Device.JAVA;
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
