package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class FloatArrayAllocator implements MemoryAllocator<float[]> {

    private static final FloatArrayAllocator INSTANCE = new FloatArrayAllocator();

    public static MemoryAllocator<float[]> instance() {
        return INSTANCE;
    }

    private FloatArrayAllocator() {
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public long defaultByteAlignment() {
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
