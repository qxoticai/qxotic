package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.Memory;

/** Minimal wrapper to preserve Mojo device identity over HIP-backed memory. */
final class MojoMemory<T> implements Memory<T> {

    private final Device device;
    private final Memory<T> delegate;

    MojoMemory(Device device, Memory<T> delegate) {
        if (delegate instanceof MojoMemory) {
            throw new IllegalArgumentException("MojoMemory delegate must be a non-Mojo memory");
        }
        this.device = device;
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
        return device;
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
