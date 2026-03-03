package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class MetalMemoryAllocator implements MemoryAllocator<MetalDevicePtr> {

    private static final MetalMemoryAllocator INSTANCE = new MetalMemoryAllocator();

    static MetalMemoryAllocator instance() {
        return INSTANCE;
    }

    private MetalMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.METAL;
    }

    @Override
    public Memory<MetalDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        MetalRuntime.requireAvailable();
        int storageMode = MetalMemoryPolicy.current().storageMode();
        long handle = MetalRuntime.malloc(byteSize, storageMode);
        return new MetalMemory(new MetalDevicePtr(handle), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
