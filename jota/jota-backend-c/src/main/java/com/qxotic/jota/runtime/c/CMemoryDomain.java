package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryAccess;
import com.qxotic.jota.runtime.nativeimpl.NativeMemoryOperations;
import java.lang.foreign.MemorySegment;

final class CMemoryDomain implements MemoryDomain<MemorySegment> {
    private final Device device;
    private final MemoryAllocator<MemorySegment> allocator;

    CMemoryDomain(Device device) {
        this.device = device;
        this.allocator = new CMemoryAllocator(device);
    }

    @Override
    public Device device() {
        return device;
    }

    @Override
    public MemoryAllocator<MemorySegment> memoryAllocator() {
        return allocator;
    }

    @Override
    public MemoryAccess<MemorySegment> directAccess() {
        return NativeMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return NativeMemoryOperations.instance();
    }

    @Override
    public void close() {
        if (allocator instanceof AutoCloseable closeable) {
            try {
                closeable.close();
            } catch (Exception e) {
                throw new IllegalStateException("Failed to close C memory allocator", e);
            }
        }
    }
}
