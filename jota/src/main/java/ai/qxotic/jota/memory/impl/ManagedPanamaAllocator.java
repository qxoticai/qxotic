package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryArena;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

final class ManagedPanamaAllocator implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private Arena arena = Arena.ofAuto();

    @Override
    public Device device() {
        return Device.NATIVE;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        return PanamaMemory.of(arena.allocate(byteSize, byteAlignment));
    }

    @Override
    public void close() {
        // arena.close() // non-closeable arena
        if (this.arena == null) {
            throw new IllegalStateException("already closed");
        } else {
            this.arena = null;
        }
    }
}
