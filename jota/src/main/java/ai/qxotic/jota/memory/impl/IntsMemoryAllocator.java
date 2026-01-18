package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class IntsMemoryAllocator implements MemoryAllocator<int[]> {

    private static final IntsMemoryAllocator INSTANCE = new IntsMemoryAllocator();

    public static MemoryAllocator<int[]> instance() {
        return INSTANCE;
    }

    private IntsMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.JAVA;
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
