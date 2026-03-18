package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;

/** Minimal allocator wrapper that returns Mojo-identified memory. */
final class MojoMemoryAllocator<T> implements MemoryAllocator<T> {

    private final MemoryAllocator<T> delegate;

    MojoMemoryAllocator(MemoryAllocator<T> delegate) {
        if (delegate instanceof MojoMemoryAllocator) {
            throw new IllegalArgumentException(
                    "MojoMemoryAllocator delegate must be a non-Mojo allocator");
        }
        this.delegate = delegate;
    }

    @Override
    public Device device() {
        return new Device(DeviceType.MOJO, 0);
    }

    @Override
    public Memory<T> allocateMemory(long byteSize, long byteAlignment) {
        return new MojoMemory<>(delegate.allocateMemory(byteSize, byteAlignment));
    }

    @Override
    public long defaultByteAlignment() {
        return delegate.defaultByteAlignment();
    }

    @Override
    public long memoryGranularity() {
        return delegate.memoryGranularity();
    }
}
