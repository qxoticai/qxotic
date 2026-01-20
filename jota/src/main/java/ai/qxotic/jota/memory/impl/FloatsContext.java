package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.*;

class FloatsContext implements MemoryContext<float[]> {

    private static final FloatsContext INSTANCE = new FloatsContext();

    static MemoryContext<float[]> instance() {
        return INSTANCE;
    }

    private FloatsContext() {}

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<float[]> memoryAllocator() {
        return FloatsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<float[]> memoryAccess() {
        return FloatsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<float[]> memoryOperations() {
        return FloatsMemoryOperations.instance();
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{float[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
