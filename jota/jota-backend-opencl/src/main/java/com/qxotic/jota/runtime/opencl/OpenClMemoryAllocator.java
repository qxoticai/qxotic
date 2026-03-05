package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class OpenClMemoryAllocator implements MemoryAllocator<OpenClDevicePtr> {

    private static final OpenClMemoryAllocator INSTANCE = new OpenClMemoryAllocator();

    static OpenClMemoryAllocator instance() {
        return INSTANCE;
    }

    private OpenClMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.OPENCL;
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
        return new OpenClMemory(new OpenClDevicePtr(handle), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
