package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.Memory;

import java.lang.foreign.MemorySegment;
import java.util.Objects;

final class PanamaMemory implements Memory<MemorySegment> {

    final MemorySegment memorySegment;

    private PanamaMemory(MemorySegment memorySegment) {
        this.memorySegment = Objects.requireNonNull(memorySegment);
    }

    static PanamaMemory of(MemorySegment memorySegment) {
        return new PanamaMemory(memorySegment);
    }

    @Override
    public long byteSize() {
        return memorySegment.byteSize();
    }

    @Override
    public boolean isReadOnly() {
        return memorySegment.isReadOnly();
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public MemorySegment base() {
        return memorySegment;
    }

    public PanamaMemory asReadOnly() {
        if (isReadOnly()) {
            return this;
        } else {
            return new PanamaMemory(this.memorySegment.asReadOnly());
        }
    }

    @Override
    public String toString() {
        return "PanamaMemory{" +
                "size=" + byteSize() +
                "readOnly=" + isReadOnly() +
                "device=" + device() +
                '}';
    }
}
