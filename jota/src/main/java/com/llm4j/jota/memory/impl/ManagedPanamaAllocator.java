package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;
import com.llm4j.jota.memory.MemoryAllocator;
import com.llm4j.jota.memory.MemoryArena;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class ManagedPanamaAllocator implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private Arena arena = Arena.ofAuto();

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        return PanamaMemory.of(arena.allocate(byteSize, byteAlignment));
    }

    @Override
    public void close() {
        if (this.arena == null) {
            throw new IllegalStateException("already closed");
        } else {
            this.arena = null;
        }
    }
}
