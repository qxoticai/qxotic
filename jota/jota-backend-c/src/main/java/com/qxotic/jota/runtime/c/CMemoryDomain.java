package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import com.qxotic.jota.runtime.panama.PanamaMemoryAccess;
import com.qxotic.jota.runtime.panama.PanamaMemoryOperations;
import java.lang.foreign.MemorySegment;

final class CMemoryDomain implements MemoryDomain<MemorySegment> {

    private final MemoryAllocator<MemorySegment> allocator = new CMemoryAllocator();

    @Override
    public Device device() {
        return Device.C;
    }

    @Override
    public MemoryAllocator<MemorySegment> memoryAllocator() {
        return allocator;
    }

    @Override
    public MemoryAccess<MemorySegment> directAccess() {
        return PanamaMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return PanamaMemoryOperations.instance();
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
