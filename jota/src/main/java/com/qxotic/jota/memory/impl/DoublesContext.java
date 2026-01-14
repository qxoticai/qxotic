package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.FloatOperations;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryContext;
import com.qxotic.jota.memory.MemoryOperations;

class DoublesContext implements MemoryContext<double[]> {

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<double[]> memoryAllocator() {
        return DoublesMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<double[]> memoryAccess() {
        return DoublesMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<double[]> memoryOperations() {
        return DoublesMemoryOperations.instance();
    }

    @Override
    public FloatOperations<double[]> floatOperations() {
        return null;
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{double[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
