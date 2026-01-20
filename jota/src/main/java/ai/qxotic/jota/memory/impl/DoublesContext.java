package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class DoublesContext implements MemoryContext<double[]> {

    private static final DoublesContext INSTANCE = new DoublesContext();

    static MemoryContext<double[]> instance() {
        return INSTANCE;
    }

    private DoublesContext() {}

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
