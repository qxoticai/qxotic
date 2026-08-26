package com.qxotic.jota.memory;

import com.qxotic.jota.memory.internal.MemoryAllocatorFactory;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;

/**
 * {@link MemoryAllocator}s. {@code of} wraps a lifetime you own; {@code new} creates one that the
 * caller owns and closes. There is no default native allocator: native memory always names its
 * {@link Arena}.
 */
public final class MemoryAllocators {

    private MemoryAllocators() {}

    /**
     * An arena over a JDK {@link Arena}: the arena's lifetime, confinement and use-after-close
     * checks apply, allocations are zero-filled, and the default alignment is 64 bytes. {@code
     * close()} delegates, so it is unsupported for {@code Arena.ofAuto()} and {@code
     * Arena.global()}. The wrapper of {@code Arena.global()} is a shared instance.
     */
    public static MemoryArena<MemorySegment> ofArena(Arena arena) {
        return MemoryAllocatorFactory.ofArena(arena);
    }

    /**
     * A new malloc-backed arena whose buffers can be freed individually ({@link
     * ScopedMemory#close()}) or all at once ({@link MemoryArena#close()}). Not zero-filled; use
     * after close is undefined. For frequent, short-lived buffers where GC-managed release is not
     * enough.
     */
    public static ScopedMemoryAllocatorArena<MemorySegment> newScopedArena() {
        return MemoryAllocatorFactory.newPanamaArena();
    }

    /** A new allocator of native-order {@link ByteBuffer}s, direct or heap. */
    public static MemoryAllocator<ByteBuffer> newByteBuffer(boolean direct) {
        return MemoryAllocatorFactory.ofByteBuffer(direct);
    }
}
