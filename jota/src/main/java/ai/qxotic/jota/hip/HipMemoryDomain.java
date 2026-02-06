package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryOperations;

public final class HipMemoryDomain implements MemoryDomain<HipDevicePtr> {

    private static final HipMemoryDomain INSTANCE = new HipMemoryDomain();

    public static HipMemoryDomain instance() {
        return INSTANCE;
    }

    private HipMemoryDomain() {}

    @Override
    public Device device() {
        return Device.HIP;
    }

    @Override
    public MemoryAllocator<HipDevicePtr> memoryAllocator() {
        return HipMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<HipDevicePtr> directAccess() {
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
        return "HipMemoryDomain{HipDevicePtr, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}
