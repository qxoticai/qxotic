package com.qxotic.jota.memory;

/** Only bulk de-allocation is supported. */
public interface MemoryArena<B> extends MemoryAllocator<B>, AutoCloseable {
    /** Releases ALL memory in the arena (invalidates all views) */
    @Override
    void close();

    /**
     * True while memory previously allocated from this arena remains valid; false once {@link
     * #close()} has invalidated it. Arenas whose buffers are GC-managed, or whose {@code close()}
     * is a no-op, always report true.
     */
    boolean isAlive();
}
