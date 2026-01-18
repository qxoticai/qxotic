package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.FloatOperations;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class BytesContext implements MemoryContext<byte[]> {

    private static final BytesContext INSTANCE = new BytesContext();

    static MemoryContext<byte[]> instance() {
        return INSTANCE;
    }

    private BytesContext() {}

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<byte[]> memoryAllocator() {
        return BytesMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<byte[]> memoryAccess() {
        return BytesMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<byte[]> memoryOperations() {
        return BytesMemoryOperations.instance();
    }

    @Override
    public FloatOperations<byte[]> floatOperations() {
        return null;
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
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
