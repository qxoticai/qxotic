package com.qxotic.jota.runtime.c;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryDomains;
import com.qxotic.jota.memory.MemoryOperations;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class CMemoryDomain implements MemoryDomain<MemorySegment> {

    /** Host-side services (access, operations); the global arena is never allocated from here. */
    private static final MemoryDomain<MemorySegment> HOST =
            MemoryDomains.of(MemoryAllocators.ofArena(Arena.global()));

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
        return HOST.directAccess();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return HOST.memoryOperations();
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
