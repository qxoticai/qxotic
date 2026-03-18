package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

final class CudaMemoryAllocator implements MemoryAllocator<CudaDevicePtr> {

    private static final CudaMemoryAllocator INSTANCE = new CudaMemoryAllocator();

    static CudaMemoryAllocator instance() {
        return INSTANCE;
    }

    private CudaMemoryAllocator() {}

    @Override
    public Device device() {
        return new Device(DeviceType.CUDA, 0);
    }

    @Override
    public Memory<CudaDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        CudaRuntime.requireAvailable();
        long ptr = CudaRuntime.malloc(byteSize);
        return new CudaMemory(new CudaDevicePtr(ptr), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
