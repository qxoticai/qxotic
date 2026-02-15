package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryDomain;
import ai.qxotic.jota.memory.MemoryOperations;

class LongsDomain implements MemoryDomain<long[]> {

    private static final Device LONGS = Device.CPU.child("longs");
    private static final LongsDomain INSTANCE = new LongsDomain();

    static MemoryDomain<long[]> instance() {
        return INSTANCE;
    }

    private LongsDomain() {}

    @Override
    public Device device() {
        return LONGS;
    }

    @Override
    public MemoryAllocator<long[]> memoryAllocator() {
        return LongsMemoryAllocator.instance();
    }

    @Override
    public MemoryAccess<long[]> directAccess() {
        return LongsMemoryAccess.instance();
    }

    @Override
    public MemoryOperations<long[]> memoryOperations() {
        return LongsMemoryOperations.instance();
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
                .append(directAccess() != null)
                .append('}')
                .toString();
    }
}
