package com.qxotic.jota.memory.internal;

import com.qxotic.jota.Device;
import com.qxotic.jota.DeviceType;
import com.qxotic.jota.memory.Memory;
import com.qxotic.jota.memory.MemoryArena;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/**
 * A {@link MemoryArena} over a caller-supplied JDK {@link Arena}: allocation, lifetime, thread
 * confinement and use-after-close checks are the arena's. {@link #close()} delegates, so it throws
 * {@link UnsupportedOperationException} for {@code Arena.ofAuto()} and {@code Arena.global()},
 * which cannot be closed. Allocations are zero-filled by the JDK.
 */
final class ArenaAllocator implements MemoryArena<MemorySegment> {

    /** The global arena is a process singleton, so its wrapper is one too. */
    private static final ArenaAllocator GLOBAL = new ArenaAllocator(Arena.global());

    private final Arena arena;

    private ArenaAllocator(Arena arena) {
        this.arena = Objects.requireNonNull(arena, "arena");
    }

    static MemoryArena<MemorySegment> of(Arena arena) {
        return arena == Arena.global() ? GLOBAL : new ArenaAllocator(arena);
    }

    @Override
    public Device device() {
        return DeviceType.PANAMA.deviceIndex(0);
    }

    @Override
    public long memoryGranularity() {
        return Byte.BYTES;
    }

    /** Kernels want cacheline/vector alignment; callers may still override per allocation. */
    @Override
    public long defaultByteAlignment() {
        return 64;
    }

    @Override
    public Memory<MemorySegment> allocateMemory(long byteSize, long byteAlignment) {
        return NativeMemorySegmentMemory.of(arena.allocate(byteSize, byteAlignment));
    }

    @Override
    public void close() {
        arena.close();
    }

    @Override
    public boolean isAlive() {
        return arena.scope().isAlive();
    }

    @Override
    public String toString() {
        return "ArenaAllocator{" + arena + ", alive=" + isAlive() + '}';
    }
}
