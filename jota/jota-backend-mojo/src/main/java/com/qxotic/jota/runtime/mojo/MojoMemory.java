package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

/** Minimal wrapper to preserve Device.MOJO identity over HIP-backed memory. */
final class MojoMemory<T> implements Memory<T> {

    private final Memory<T> delegate;

    MojoMemory(Memory<T> delegate) {
        if (delegate instanceof MojoMemory) {
            throw new IllegalArgumentException("MojoMemory delegate must be a non-Mojo memory");
        }
        this.delegate = delegate;
    }

    Memory<T> delegate() {
        return delegate;
    }

    @Override
    public long byteSize() {
        return delegate.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return delegate.isReadOnly();
    }

    @Override
    public Device device() {
        return Device.MOJO;
    }

    @Override
    public T base() {
        return delegate.base();
    }

    @Override
    public long memoryGranularity() {
        return delegate.memoryGranularity();
    }
}
