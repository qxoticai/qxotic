package com.qxotic.jota.memory;

/**
 * An arena whose buffers can also be freed one by one; see {@link
 * MemoryAllocators#newScopedArena()}.
 */
public interface ScopedArena<B> extends ScopedMemoryAllocator<B>, MemoryArena<B> {}
