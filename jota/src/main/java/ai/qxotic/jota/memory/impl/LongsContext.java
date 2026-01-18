package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.FloatOperations;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryContext;
import ai.qxotic.jota.memory.MemoryOperations;

class LongsContext implements MemoryContext<long[]> {

    private static final LongsContext INSTANCE = new LongsContext();

    static MemoryContext<long[]> instance() {
        return INSTANCE;
    }

    private LongsContext() {}

    @Override
    public Device device() {
        return Device.JAVA;
    }

    @Override
    public MemoryAllocator<long[]> memoryAllocator() {
        return LongsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<long[]> memoryAccess() {
        return LongsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<long[]> memoryOperations() {
        return LongsMemoryOperations.instance();
    }

    @Override
    public FloatOperations<long[]> floatOperations() {
        return null;
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
                .append(memoryAccess() != null)
                .append('}')
                .toString();
    }
}
