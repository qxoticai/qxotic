package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class LongsMemoryAllocator implements MemoryAllocator<long[]> {

    private static final LongsMemoryAllocator INSTANCE = new LongsMemoryAllocator();

    public static MemoryAllocator<long[]> instance() {
        return INSTANCE;
    }

    private LongsMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.PANAMA;
    }

    @Override
    public long defaultByteAlignment() {
        return Long.BYTES;
    }

    @Override
    public long memoryGranularity() {
        return Long.BYTES;
    }

    @Override
    public Memory<long[]> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        if (byteSize % Long.BYTES != 0) {
            throw new IllegalArgumentException("unaligned (8) byteSize");
        }
        int length = Math.toIntExact(byteSize / Long.BYTES);
        return MemoryFactory.ofLongs(new long[length]);
    }
}
