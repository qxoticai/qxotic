package ai.qxotic.jota.memory.impl;

import ai.qxotic.jota.Device;
import ai.qxotic.jota.memory.Memory;
import ai.qxotic.jota.memory.MemoryAllocator;
import ai.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;

final class PanamaAutoAllocator
        implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private final AtomicReference<Arena> arenaRef = new AtomicReference<>(Arena.ofAuto());

    @Override
    public Device device() {
        return Device.PANAMA;
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        Arena arena = getArena();
        return PanamaMemory.of(arena.allocate(byteSize, byteAlignment));
    }

    private Arena getArena() {
        Arena arena = arenaRef.get();
        if (arena == null) {
            throw new IllegalStateException("arena already closed");
        }
        return arena;
    }

    @Override
    public void close() {
        Arena oldArena = arenaRef.getAndSet(null);
        if (oldArena == null) {
            throw new IllegalStateException("arena already closed");
        }
    }
}
