package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.FloatOperations;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryOperations;

class BooleansContext implements MemoryContext<boolean[]> {

    private static final BooleansContext INSTANCE = new BooleansContext();

    static MemoryContext<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansContext() {
    }

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<boolean[]> memoryAllocator() {
        return BooleansMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<boolean[]> memoryAccess() {
        return BooleansMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<boolean[]> memoryOperations() {
        return BooleansMemoryOperations.instance();
    }

    @Override
    public FloatOperations<boolean[]> floatOperations() {
        return null;  // No float operations for boolean[]
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{boolean[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
