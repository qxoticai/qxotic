package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.FloatOperations;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryOperations;

class BytesContext implements MemoryContext<byte[]> {

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<byte[]> memoryAllocator() {
        return BytesMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<byte[]> memoryAccess() {
        return BytesMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<byte[]> memoryOperations() {
        return BytesMemoryOperations.instance();
    }

    @Override
    public FloatOperations<byte[]> floatOperations() {
        return null;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{byte[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
