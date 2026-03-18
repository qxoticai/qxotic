package com.qxotic.jota.runtime.cuda;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import java.lang.ref.Cleaner;

final class CudaMemory implements Memory<CudaDevicePtr> {

    private static final Cleaner CLEANER = Cleaner.create();

    private final CudaDevicePtr handle;
    private final long byteSize;
    private final Cleaner.Cleanable cleanable;

    CudaMemory(CudaDevicePtr handle, long byteSize) {
        this.handle = handle;
        this.byteSize = byteSize;
        this.cleanable = CLEANER.register(this, new Releaser(handle.address()));
    }

    @Override
    public long byteSize() {
        return byteSize;
    }

    @Override
    public boolean isReadOnly() {
        return false;
    }

    @Override
    public Device device() {
        return new Device(DeviceType.CUDA, 0);
    }

    @Override
    public CudaDevicePtr base() {
        return handle;
    }

    @Override
    public long memoryGranularity() {
        return 1;
    }

    private static final class Releaser implements Runnable {
        private final long handle;

        private Releaser(long handle) {
            this.handle = handle;
        }

        @Override
        public void run() {
            if (handle != 0L && CudaRuntime.isAvailable()) {
                CudaRuntime.free(handle);
            }
        }
    }
}
