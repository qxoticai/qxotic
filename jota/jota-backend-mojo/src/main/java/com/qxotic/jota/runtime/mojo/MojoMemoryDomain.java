package com.qxotic.jota.runtime.mojo;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.memory.MemoryView;

final class MojoMemoryDomain<T> implements MemoryDomain<T> {

    private final MemoryDomain<T> delegate;
    private final MojoMemoryAllocator<T> allocator;

    MojoMemoryDomain(MemoryDomain<T> delegate) {
        if (delegate instanceof MojoMemoryDomain) {
            throw new IllegalArgumentException(
                    "MojoMemoryDomain delegate must be a non-Mojo domain");
        }
        this.delegate = delegate;
        this.allocator = new MojoMemoryAllocator<>(delegate.memoryAllocator());
    }

    @Override
    public Device device() {
        return Device.MOJO;
    }

    @Override
    public MemoryAllocator<T> memoryAllocator() {
        return allocator;
    }

    @Override
    public MemoryAccess<T> directAccess() {
        return delegate.directAccess();
    }

    @Override
    public MemoryOperations<T> memoryOperations() {
        return delegate.memoryOperations();
    }

    @Override
    public void copy(MemoryView<T> src, MemoryView<T> dst) {
        MemoryView<T> srcView = unwrap(src);
        MemoryView<T> dstView = unwrap(dst);
        delegate.copy(srcView, dstView);
    }

    private static <T> MemoryView<T> unwrap(MemoryView<T> view) {
        if (view.memory() instanceof MojoMemory<?> mojoMemory) {
            @SuppressWarnings("unchecked")
            MemoryView<T> unwrapped =
                    MemoryView.of(
                            (com.qxotic.jota.memory.Memory<T>) mojoMemory.delegate(),
                            view.byteOffset(),
                            view.dataType(),
                            view.layout());
            return unwrapped;
        }
        return view;
    }

    @Override
    public void close() {
        delegate.close();
    }
}
