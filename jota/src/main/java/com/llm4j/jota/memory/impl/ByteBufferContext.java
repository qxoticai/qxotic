package com.llm4j.jota.memory.impl;

import com.llm4j.jota.Device;
import com.llm4j.jota.memory.*;

import java.nio.ByteBuffer;
import java.util.Objects;

class ByteBufferContext implements Context<ByteBuffer> {

    private final FloatOperations<ByteBuffer> floatOperations = new GenericFloatOperations<>(ByteBufferMemoryAccess.instance());
    private final MemoryAllocator<ByteBuffer> memoryAllocator;

    ByteBufferContext(MemoryAllocator<ByteBuffer> memoryAllocator) {
        this.memoryAllocator = Objects.requireNonNull(memoryAllocator);
    }

    @Override
    public Device device() {
        return Device.CPU;
    }

    @Override
    public MemoryAllocator<ByteBuffer> memoryAllocator() {
        return memoryAllocator;
    }

    @Override
    public MemoryAccess<ByteBuffer> memoryAccess() {
        return ByteBufferMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<ByteBuffer> memoryOperations() {
        return ByteBufferMemoryOperations.instance();
    }

    @Override
    public FloatOperations<ByteBuffer> floatOperations() {
        return this.floatOperations;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return "context ByteBuffer";
    }
}
