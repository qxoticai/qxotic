package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;
import java.nio.ByteBuffer;
import java.util.Objects;

record ByteBufferDomain(MemoryAllocator<ByteBuffer> memoryAllocator)
        implements MemoryDomain<ByteBuffer> {

    ByteBufferDomain(MemoryAllocator<ByteBuffer> memoryAllocator) {
        this.memoryAllocator = Objects.requireNonNull(memoryAllocator);
    }

    @Override
    public Device device() {
        return memoryAllocator.device();
    }

    @Override
    public MemoryAccess<ByteBuffer> directAccess() {
        return ByteBufferMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<ByteBuffer> memoryOperations() {
        return ByteBufferMemoryOperations.instance();
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return "Context{ByteBuffer, device="
                + device()
                + ", directAccess="
                + (directAccess() != null)
                + '}';
    }
}
