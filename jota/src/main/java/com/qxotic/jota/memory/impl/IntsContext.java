package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.FloatOperations;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryOperations;

class IntsContext implements MemoryContext<int[]> {

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<int[]> memoryAllocator() {
        return IntsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<int[]> memoryAccess() {
        return IntsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<int[]> memoryOperations() {
        return IntsMemoryOperations.instance();
    }

    @Override
    public FloatOperations<int[]> floatOperations() {
        return null;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{int[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
