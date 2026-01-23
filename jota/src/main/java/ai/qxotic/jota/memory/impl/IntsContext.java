package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class IntsContext implements MemoryContext<int[]> {

    private static final Device INTS = Device.CPU.child("ints");
    private static final IntsContext INSTANCE = new IntsContext();

    static MemoryContext<int[]> instance() {
        return INSTANCE;
    }

    private IntsContext() {}

    @Override
    public Device device() {
        return INTS;
    }

    @Override
    public MemoryAllocator<int[]> memoryAllocator() {
        return IntsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<int[]> memoryAccess() {
        return IntsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<int[]> memoryOperations() {
        return IntsMemoryOperations.instance();
    }

    @Override
    public void close() {
        // Nothing to do, memory is managed by the GC.
    }

    @Override
    public String toString() {
        return new StringBuilder("Context{int[], device=")
                .append(device())
                .append(", directAccess=")
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
