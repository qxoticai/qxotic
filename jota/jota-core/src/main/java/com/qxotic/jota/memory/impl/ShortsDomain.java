package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class ShortsDomain implements MemoryDomain<short[]> {

    private static final ShortsDomain INSTANCE = new ShortsDomain();

    static MemoryDomain<short[]> instance() {
        return INSTANCE;
    }

    private ShortsDomain() {}

    @Override
    public Device device() {
        return DeviceType.JAVA.deviceIndex(0);
    }

    @Override
    public MemoryAllocator<short[]> memoryAllocator() {
        return ShortsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<short[]> directAccess() {
        return ShortsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<short[]> memoryOperations() {
        return ShortsMemoryOperations.instance();
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{short[], device=")
                .append(device())
                .append(", directAccess=")
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
