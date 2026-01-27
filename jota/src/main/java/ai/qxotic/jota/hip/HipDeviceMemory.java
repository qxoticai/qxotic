package ai.qxotic.jota.hip;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import java.lang.ref.Cleaner;

final class HipDeviceMemory implements Memory<HipDevicePtr> {

    private static final Cleaner CLEANER = Cleaner.create();

    private final HipDevicePtr handle;
    private final long byteSize;
    private final Cleaner.Cleanable cleanable;

    HipDeviceMemory(HipDevicePtr handle, long byteSize) {
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
        return Device.HIP;
    }

    @Override
    public HipDevicePtr base() {
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
            if (handle != 0L && HipRuntime.isAvailable()) {
                HipRuntime.free(handle);
            }
        }
    }
}
