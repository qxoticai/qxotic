package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class LongsDomain implements MemoryDomain<long[]> {

    private static final LongsDomain INSTANCE = new LongsDomain();

    static MemoryDomain<long[]> instance() {
        return INSTANCE;
    }

    private LongsDomain() {}

    @Override
    public Device device() {
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public MemoryAllocator<long[]> memoryAllocator() {
        return LongsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<long[]> directAccess() {
        return LongsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<long[]> memoryOperations() {
        return LongsMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
