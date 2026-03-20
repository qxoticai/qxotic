package com.qxotic.jota.runtime.opencl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import java.lang.ref.Cleaner;

final class OpenClMemory implements Memory<OpenClDevicePtr> {

    private static final Cleaner CLEANER = Cleaner.create();

    private final OpenClDevicePtr base;
    private final Device device;
    private final long byteSize;
    private final Cleaner.Cleanable cleanable;

    OpenClMemory(Device device, OpenClDevicePtr base, long byteSize) {
        this.device = device;
        this.base = base;
        this.byteSize = byteSize;
        this.cleanable = CLEANER.register(this, new Releaser(base.handle()));
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
        return device;
    }

    @Override
    public OpenClDevicePtr base() {
        return base;
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
            if (handle != 0L && OpenClRuntime.isAvailable()) {
                OpenClRuntime.free(handle);
            }
        }
    }
}
