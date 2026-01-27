package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

public final class HipMemoryContext implements MemoryContext<HipDevicePtr> {

    private static final HipMemoryContext INSTANCE = new HipMemoryContext();

    public static HipMemoryContext instance() {
        return INSTANCE;
    }

    private HipMemoryContext() {
    }

    @Override
    public Device device() {
        return Device.HIP;
    }

    @Override
    public MemoryAllocator<HipDevicePtr> memoryAllocator() {
        return HipMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<HipDevicePtr> memoryAccess() {
        return null;
    }

    @Override
    public MemoryOperations<HipDevicePtr> memoryOperations() {
        return HipMemoryOperations.instance();
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public String toString() {
        return "HipMemoryContext{HipDevicePtr, device=" + device() +
                ", directAccess=" + (memoryAccess() != null) +
                '}';
    }
}
