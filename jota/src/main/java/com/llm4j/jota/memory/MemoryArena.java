package com.llm4j.jota.memory;

/**
 * Only bulk de-allocation is supported.
 */
public interface MemoryArena<B> extends MemoryAllocator<B>, AutoCloseable {
    /**
     * Releases ALL memory in the arena (invalidates all views)
     */
    @Override
    void close();
}
