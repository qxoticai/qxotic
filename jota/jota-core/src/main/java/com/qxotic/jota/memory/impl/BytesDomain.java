package com.qxotic.jota.memory.impl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.MemoryAccess;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryDomain;
import com.qxotic.jota.memory.MemoryOperations;

class BytesDomain implements MemoryDomain<byte[]> {

    private static final BytesDomain INSTANCE = new BytesDomain();

    static MemoryDomain<byte[]> instance() {
        return INSTANCE;
    }

    private BytesDomain() {}

    @Override
    public Device device() {
        return new Device(DeviceType.JAVA, 0);
    }

    @Override
    public MemoryAllocator<byte[]> memoryAllocator() {
        return BytesMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<byte[]> directAccess() {
        return BytesMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<byte[]> memoryOperations() {
        return BytesMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
