package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.*;

import java.lang.foreign.MemorySegment;

final class PanamaContext implements Context<MemorySegment> {

    private final FloatOperations<MemorySegment> rawOperations = new PanamaFloatOperations(PanamaMemoryAccess.instance());
    private final ScopedMemoryAllocatorArena<MemorySegment> allocatorArena = MemoryAllocatorFactory.newPanamaArena();

    @Override
    public Device device() {
        return Device.CPU;
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
        return "context MemorySegment";
    }
}
