package com.qxotic.jota.memory;

/**
 * An allocator with a lifetime: {@link #close()} releases every allocation at once if the arena
 * owns them ({@code adopt}, {@code new}), and does nothing if it merely borrows them ({@code of}).
 * A {@link ScopedArena} additionally frees buffers one by one.
 */
public interface MemoryArena<B> extends MemoryAllocator<B>, AutoCloseable {
    /** Releases every allocation of this arena; views over them are invalid afterwards. */
    @Override
    void close();

    /**
     * True while memory previously allocated from this arena remains valid; false once {@link
     * #close()} has invalidated it. Arenas whose buffers are GC-managed, or whose {@code close()}
     * is a no-op, always report true.
     */
    boolean isAlive();
}
