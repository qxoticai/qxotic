package ai.qxotic.jota.runtime.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;

final class HipMemoryAllocator implements MemoryAllocator<HipDevicePtr> {

    private static final HipMemoryAllocator INSTANCE = new HipMemoryAllocator();

    static HipMemoryAllocator instance() {
        return INSTANCE;
    }

    private HipMemoryAllocator() {}

    @Override
    public Device device() {
        return Device.HIP;
    }

    @Override
    public Memory<HipDevicePtr> allocateMemory(long byteSize, long byteAlignment) {
        if (byteSize < 0) {
            throw new IllegalArgumentException("Negative size");
        }
        HipRuntime.requireAvailable();
        long ptr = HipRuntime.malloc(byteSize);
        return new HipMemory(new HipDevicePtr(ptr), byteSize);
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }
}
