package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

public final class OpenClMemoryDomain implements MemoryDomain<OpenClDevicePtr> {

    private static final OpenClMemoryDomain INSTANCE = new OpenClMemoryDomain();

    public static OpenClMemoryDomain instance() {
        return INSTANCE;
    }

    private OpenClMemoryDomain() {}

    @Override
    public Device device() {
        return new Device(DeviceType.OPENCL, 0);
    }

    @Override
    public MemoryAllocator<OpenClDevicePtr> memoryAllocator() {
        return OpenClMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<OpenClDevicePtr> directAccess() {
        return null;
    }

    @Override
    public MemoryOperations<OpenClDevicePtr> memoryOperations() {
        return OpenClMemoryOperations.instance();
    }

    @Override
    public void copy(MemoryView<OpenClDevicePtr> src, MemoryView<OpenClDevicePtr> dst) {
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
        OpenClStridedCopy.copy(src, dst);
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public String toString() {
        return "OpenClMemoryDomain{OpenClDevicePtr, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}
