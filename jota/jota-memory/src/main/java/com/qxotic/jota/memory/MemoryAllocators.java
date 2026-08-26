package com.qxotic.jota.memory;

import com.qxotic.jota.memory.internal.MemoryAllocatorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * {@link MemoryAllocator}s. {@code of} borrows a lifetime its creator closes; {@code adopt} takes
 * one over and closes it; {@code new} creates one the caller owns. There is no default native
 * allocator: native memory always names its {@link Arena}.
 */
public final class MemoryAllocators {

    private MemoryAllocators() {}

    /**
     * Borrows a JDK {@link Arena}: its confinement and use-after-close checks apply, allocations
     * are zero-filled, the default alignment is 64 bytes, and {@code close()} is a no-op because
     * whoever created the arena closes it. The wrapper of {@code Arena.global()} is a shared
     * instance. This is the right choice for {@code Arena.ofAuto()} and {@code Arena.global()},
     * which cannot be closed at all.
     */
    public static MemoryArena<MemorySegment> ofArena(Arena arena) {
        return MemoryAllocatorFactory.ofArena(arena);
    }

    /**
     * Adopts a JDK {@link Arena}: as {@link #ofArena}, but {@code close()} closes the arena. Adopt
     * only arenas you may close; adopting {@code Arena.ofAuto()} or {@code Arena.global()} makes
     * {@code close()} throw {@link UnsupportedOperationException}.
     */
    public static MemoryArena<MemorySegment> adoptArena(Arena arena) {
        return MemoryAllocatorFactory.adoptArena(arena);
    }

    /**
     * A new malloc-backed arena whose buffers can be freed individually ({@link
     * ScopedMemory#close()}) or all at once ({@link MemoryArena#close()}). Not zero-filled; use
     * after close is undefined. For frequent, short-lived buffers where GC-managed release is not
     * enough.
     */
    public static ScopedArena<MemorySegment> newScopedArena() {
        return MemoryAllocatorFactory.newScopedArena();
    }

    /** A new allocator of native-order {@link ByteBuffer}s, direct or heap. */
    public static MemoryAllocator<ByteBuffer> newByteBuffer(boolean direct) {
        return MemoryAllocatorFactory.ofByteBuffer(direct);
    }
}
