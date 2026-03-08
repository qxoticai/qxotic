package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class IntsDomain implements MemoryDomain<int[]> {

    private static final IntsDomain INSTANCE = new IntsDomain();

    static MemoryDomain<int[]> instance() {
        return INSTANCE;
    }

    private IntsDomain() {}

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<int[]> memoryAllocator() {
        return IntsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<int[]> directAccess() {
        return IntsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<int[]> memoryOperations() {
        return IntsMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
