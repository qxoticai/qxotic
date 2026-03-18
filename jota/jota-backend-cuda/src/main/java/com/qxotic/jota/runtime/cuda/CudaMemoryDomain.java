package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

public final class CudaMemoryDomain implements MemoryDomain<CudaDevicePtr> {

    private static final CudaMemoryDomain INSTANCE = new CudaMemoryDomain();

    public static CudaMemoryDomain instance() {
        return INSTANCE;
    }

    private CudaMemoryDomain() {}

    @Override
    public Device device() {
        return new Device(DeviceType.CUDA, 0);
    }

    @Override
    public MemoryAllocator<CudaDevicePtr> memoryAllocator() {
        return CudaMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<CudaDevicePtr> directAccess() {
        return null;
    }

    @Override
    public MemoryOperations<CudaDevicePtr> memoryOperations() {
        return CudaMemoryOperations.instance();
    }

    @Override
    public void copy(MemoryView<CudaDevicePtr> src, MemoryView<CudaDevicePtr> dst) {
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
        CudaStridedCopy.copy(src, dst);
    }

    @Override
    public void close() {
        // no-op
    }

    @Override
    public String toString() {
        return "CudaMemoryDomain{CudaDevicePtr, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}
