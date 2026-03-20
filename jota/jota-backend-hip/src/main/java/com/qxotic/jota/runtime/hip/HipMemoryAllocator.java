package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class HipMemoryAllocator implements MemoryAllocator<HipDevicePtr> {
    private final Device device;

    HipMemoryAllocator(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public Memory<HipDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        HipRuntime.requireAvailable();
        long ptr = HipRuntime.malloc(byteSize);
        return new HipMemory(device, new HipDevicePtr(ptr), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
