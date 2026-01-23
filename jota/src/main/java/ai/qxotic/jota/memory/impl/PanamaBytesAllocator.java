package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.MemorySegment;

final class PanamaBytesAllocator
        implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private static final MemoryAllocator<MemorySegment> INSTANCE = new PanamaBytesAllocator();

    public static MemoryAllocator<MemorySegment> instance() {
        return INSTANCE;
    }

    @Override
    public Device device() {
        return Device.PANAMA;
    } // JAVA?

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        if (defaultByteAlignment() % byteAlignment != 0) {
            // Cannot guarantee more than 1 alignment.
            throw new IllegalArgumentException("unsupported byteAlignment");
        }
        int byteSizeInt = Math.toIntExact(byteSize);
        return PanamaMemory.of(MemorySegment.ofArray(new byte[byteSizeInt]));
    }

    @Override
    public void close() {
        // nop
    }
}
