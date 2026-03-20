package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class OpenClMemoryAllocator implements MemoryAllocator<OpenClDevicePtr> {
    private final Device device;

    OpenClMemoryAllocator(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public Memory<OpenClDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        OpenClRuntime.requireAvailable();
        int storageMode = OpenClMemoryPolicy.current().storageMode();
        long handle = OpenClRuntime.malloc(byteSize, storageMode);
        if (handle == 0L) {
            throw new IllegalStateException("OpenCL allocation returned null handle");
        }
        return new OpenClMemory(device, new OpenClDevicePtr(handle), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
