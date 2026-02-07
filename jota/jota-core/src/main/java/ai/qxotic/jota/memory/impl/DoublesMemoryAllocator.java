package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class DoublesMemoryAllocator implements MemoryAllocator<double[]> {

    private static final DoublesMemoryAllocator INSTANCE = new DoublesMemoryAllocator();

    public static MemoryAllocator<double[]> instance() {
        return INSTANCE;
    }

    private DoublesMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.PANAMA;
    }

    @Override
    public long defaultByteAlignment() {
        return Double.BYTES;
    }

    @Override
    public long memoryGranularity() {
        return Double.BYTES;
    }

    @Override
    public Memory<double[]> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        if (byteSize % memoryGranularity() != 0) {
            throw new IllegalArgumentException("unaligned byteSize");
        }
        int length = Math.toIntExact(byteSize / Double.BYTES);
        return MemoryFactory.ofDoubles(new double[length]);
    }
}
