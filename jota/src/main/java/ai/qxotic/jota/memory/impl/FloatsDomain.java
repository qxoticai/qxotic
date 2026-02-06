package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryOperations;

class FloatsDomain implements MemoryDomain<float[]> {

    private static final Device FLOATS = Device.CPU.child("floats");
    private static final FloatsDomain INSTANCE = new FloatsDomain();

    static MemoryDomain<float[]> instance() {
        return INSTANCE;
    }

    private FloatsDomain() {}

    @Override
    public Device device() {
        return FLOATS;
    }

    @Override
    public MemoryAllocator<float[]> memoryAllocator() {
        return FloatsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<float[]> directAccess() {
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
