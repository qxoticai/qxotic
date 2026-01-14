package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.*;

import java.lang.foreign.MemorySegment;

final class PanamaContext implements MemoryContext<MemorySegment> {

    private final FloatOperations<MemorySegment> rawOperations = new PanamaFloatOperations(PanamaMemoryAccess.instance());
    private final ScopedMemoryAllocatorArena<MemorySegment> allocatorArena = MemoryAllocatorFactory.newPanamaArena();

    @Override
    public Device device() {
        return Device.NATIVE;
    }

    @Override
    public ScopedMemoryAllocatorArena<MemorySegment> memoryAllocator() {
        return allocatorArena;
    }

    @Override
    public MemoryAccess<MemorySegment> memoryAccess() {
        return PanamaMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<MemorySegment> memoryOperations() {
        return PanamaMemoryOperations.instance();
    }

    @Override
    public FloatOperations<MemorySegment> floatOperations() {
        return this.rawOperations;
    }

    @Override
    public void close() {
        allocatorArena.close();
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{MemorySegment, device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
