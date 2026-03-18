package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class DoublesDomain implements MemoryDomain<double[]> {

    private static final DoublesDomain INSTANCE = new DoublesDomain();

    static MemoryDomain<double[]> instance() {
        return INSTANCE;
    }

    private DoublesDomain() {}

    @Override
    public Device device() {
        return new Device(DeviceType.JAVA, 0);
    }

    @Override
    public MemoryAllocator<double[]> memoryAllocator() {
        return DoublesMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<double[]> directAccess() {
        return DoublesMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<double[]> memoryOperations() {
        return DoublesMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
