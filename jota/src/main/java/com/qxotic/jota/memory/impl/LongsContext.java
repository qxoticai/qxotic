package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.FloatOperations;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryOperations;

class LongsContext implements MemoryContext<long[]> {

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<long[]> memoryAllocator() {
        return LongsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<long[]> memoryAccess() {
        return LongsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<long[]> memoryOperations() {
        return LongsMemoryOperations.instance();
    }

    @Override
    public FloatOperations<long[]> floatOperations() {
        return null;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{long[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
