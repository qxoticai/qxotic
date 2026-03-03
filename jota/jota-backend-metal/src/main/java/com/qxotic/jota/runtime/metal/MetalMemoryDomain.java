package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

public final class MetalMemoryDomain implements MemoryDomain<MetalDevicePtr> {

    private static final MetalMemoryDomain INSTANCE = new MetalMemoryDomain();

    public static MetalMemoryDomain instance() {
        return INSTANCE;
    }

    private MetalMemoryDomain() {}

    @Override
    public Device device() {
        return Device.METAL;
    }

    @Override
    public MemoryAllocator<MetalDevicePtr> memoryAllocator() {
        return MetalMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<MetalDevicePtr> directAccess() {
        return null;
    }

    @Override
    public MemoryOperations<MetalDevicePtr> memoryOperations() {
        return MetalMemoryOperations.instance();
    }

    @Override
    public void copy(MemoryView<MetalDevicePtr> src, MemoryView<MetalDevicePtr> dst) {
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
        MetalStridedCopy.copy(src, dst);
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public String toString() {
        return "MetalMemoryDomain{MetalDevicePtr, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}
