package com.qxotic.jota.runtime.metal;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;
import java.lang.ref.Cleaner;

final class MetalMemory implements Memory<MetalDevicePtr> {

    private static final Cleaner CLEANER = Cleaner.create();

    private final MetalDevicePtr base;
    private final long byteSize;
    private final Cleaner.Cleanable cleanable;

    MetalMemory(MetalDevicePtr base, long byteSize) {
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
        return Device.METAL;
    }

    @Override
    public MetalDevicePtr base() {
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
            if (handle != 0L && MetalRuntime.isAvailable()) {
                MetalRuntime.free(handle);
            }
        }
    }
}
