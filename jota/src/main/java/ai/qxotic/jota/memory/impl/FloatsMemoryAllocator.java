package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class FloatsMemoryAllocator implements MemoryAllocator<float[]> {

    private static final FloatsMemoryAllocator INSTANCE = new FloatsMemoryAllocator();

    public static MemoryAllocator<float[]> instance() {
        return INSTANCE;
    }

    private FloatsMemoryAllocator() {
    }

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public long defaultByteAlignment() {
        return Float.BYTES;
    }

    @Override
    public long memoryGranularity() {
        return Float.BYTES;
    }

    @Override
    public Memory<float[]> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            // Cannot guarantee more than Float.BYTES alignment.
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        if (byteSize % Float.BYTES != 0) {
            throw new IllegalArgumentException("unaligned (4) byteSize");
        }
        int length = Math.toIntExact(byteSize / Float.BYTES);
        return MemoryFactory.ofFloats(new float[length]);
    }
}
