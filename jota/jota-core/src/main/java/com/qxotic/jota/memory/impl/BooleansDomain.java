package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class BooleansDomain implements MemoryDomain<boolean[]> {

    private static final Device BOOLEANS = Device.CPU.child("booleans");
    private static final BooleansDomain INSTANCE = new BooleansDomain();

    static MemoryDomain<boolean[]> instance() {
        return INSTANCE;
    }

    private BooleansDomain() {}

    @Override
    public Device device() {
        return BOOLEANS;
    }

    @Override
    public MemoryAllocator<boolean[]> memoryAllocator() {
        return BooleansMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<boolean[]> directAccess() {
        return BooleansMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<boolean[]> memoryOperations() {
        return BooleansMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
