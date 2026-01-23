package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class FloatsContext implements MemoryContext<float[]> {

    private static final Device FLOATS = Device.CPU.child("floats");
    private static final FloatsContext INSTANCE = new FloatsContext();


    static MemoryContext<float[]> instance() {
        return INSTANCE;
    }

    private FloatsContext() {
    }

    @Override
    public Device device() {
        return FLOATS;
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
