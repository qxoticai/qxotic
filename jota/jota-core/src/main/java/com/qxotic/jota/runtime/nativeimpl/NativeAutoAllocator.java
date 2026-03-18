package com.qxotic.jota.runtime.nativeimpl;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicReference;

final class NativeAutoAllocator
        implements MemoryAllocator<MemorySegment>, MemoryArena<MemorySegment> {

    private final AtomicReference<Arena> arenaRef = new AtomicReference<>(Arena.ofAuto());

    @Override
    public Device device() {
        return new Device(DeviceType.PANAMA, 0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        Arena arena = getArena();
        return NativeMemorySegmentMemory.of(arena.allocate(byteSize, byteAlignment));
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
