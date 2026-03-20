package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

public final class OpenClMemoryDomain implements MemoryDomain<OpenClDevicePtr> {
    public static OpenClMemoryDomain instance() {
        String token = System.getProperty(OpenClRuntime.DEVICE_INDEX_PROPERTY, "0");
        int index;
        try {
            index = Integer.parseInt(token.trim());
        } catch (NumberFormatException ignored) {
            index = 0;
        }
        return new OpenClMemoryDomain(DeviceType.OPENCL.deviceIndex(Math.max(index, 0)));
    }

    private final Device device;
    private final MemoryAllocator<OpenClDevicePtr> allocator;

    public OpenClMemoryDomain(Device device) {
        this.device = device;
        this.allocator = new OpenClMemoryAllocator(device);
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryAllocator<OpenClDevicePtr> memoryAllocator() {
        return allocator;
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
