package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.FloatOperations;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class ShortsContext implements MemoryContext<short[]> {

    private static final ShortsContext INSTANCE = new ShortsContext();

    static MemoryContext<short[]> instance() {
        return INSTANCE;
    }

    private ShortsContext() {
    }

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<short[]> memoryAllocator() {
        return ShortsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<short[]> memoryAccess() {
        return ShortsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<short[]> memoryOperations() {
        return ShortsMemoryOperations.instance();
    }

    @Override
    public FloatOperations<short[]> floatOperations() {
        return null;
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
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
