package com.qxotic.jota.runtime.hip;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

public final class HipMemoryDomain implements MemoryDomain<HipDevicePtr> {
    public static HipMemoryDomain instance() {
        return new HipMemoryDomain(DeviceType.HIP.deviceIndex(HipRuntime.currentDevice()));
    }

    private final Device device;
    private final MemoryAllocator<HipDevicePtr> allocator;

    public HipMemoryDomain(Device device) {
        this.device = device;
        this.allocator = new HipMemoryAllocator(device);
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryAllocator<HipDevicePtr> memoryAllocator() {
        return allocator;
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
    public void copy(MemoryView<HipDevicePtr> src, MemoryView<HipDevicePtr> dst) {
        if (src.isRowMajorContiguous() && dst.isRowMajorContiguous()) {
            long bytes = src.shape().size() * src.dataType().byteSize();
            if (bytes > 0) {
                memoryOperations()
                        .copy(
                                src.memory(),
                                src.byteOffset(),
                                dst.memory(),
                                dst.byteOffset(),
                                bytes);
            }
            return;
        }
        HipStridedCopy.copy(src, dst);
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
