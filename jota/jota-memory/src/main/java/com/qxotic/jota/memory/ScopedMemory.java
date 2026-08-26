package com.qxotic.jota.memory;

/**
 * Memory that can be freed on its own, ahead of its arena. Closing twice throws {@link
 * IllegalStateException}; use after close is undefined.
 */
public interface ScopedMemory<B> extends Memory<B>, AutoCloseable {
    @Override
    void close();
}
