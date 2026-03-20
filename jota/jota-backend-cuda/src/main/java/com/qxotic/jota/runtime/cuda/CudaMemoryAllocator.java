package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class CudaMemoryAllocator implements MemoryAllocator<CudaDevicePtr> {
    private final Device device;

    CudaMemoryAllocator(Device device) {
        this.device = device;
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public Memory<CudaDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        CudaRuntime.requireAvailable();
        long ptr = CudaRuntime.malloc(byteSize);
        return new CudaMemory(device, new CudaDevicePtr(ptr), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
