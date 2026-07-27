package com.qxotic.jinfer.cache;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.IdentityHashMap;

/**
 * Opaque payload storage for the prompt cache — a single-implementation seam so alternative
 * backends (in-memory arena, mmap pool, future network-attached store) can plug in without touching
 * cache logic.
 *
 * <p>Blob lifecycle: allocated → filled once → freed. Immutable after fill. Single-threaded by
 * design (only the generation worker).
 */
public interface CacheStore extends AutoCloseable {

    /** Allocates a zero-filled writable blob of {@code bytes}. */
    MemorySegment allocate(long bytes);

    /** Releases a blob back to the store. */
    void free(MemorySegment blob);

    /** Total live bytes (for budget enforcement). May be read from handler threads. */
    long usedBytes();

    @Override
    default void close() {}

    /**
     * Default backend: one shared arena per blob, closed DETERMINISTICALLY - {@link #free} and
     * {@link #close} return the memory to the OS immediately (the budget is exact, not soft by a GC
     * cycle). Safe because the blob lifecycle is single-threaded by contract: nothing reads a blob
     * after free. A Cleaner backstop covers an abandoned (dropped, unclosed) store: its remaining
     * blobs degrade to GC-eventually instead of leaking - shared arenas have no Cleaner of their
     * own.
     */
    static CacheStore inMemory() {
        final IdentityHashMap<MemorySegment, Arena> blobs = new IdentityHashMap<>();
        final Runnable sweep =
                () -> {
                    synchronized (blobs) {
                        blobs.values().forEach(Arena::close);
                        blobs.clear();
                    }
                };
        final class Store implements CacheStore {
            private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();
            private final java.lang.ref.Cleaner.Cleanable backstop = CLEANER.register(this, sweep);
            private volatile long used;

            @Override
            public MemorySegment allocate(long bytes) {
                Arena arena = Arena.ofShared();
                MemorySegment blob = arena.allocate(bytes, 64);
                synchronized (blobs) {
                    blobs.put(blob, arena);
                }
                used += bytes;
                return blob;
            }

            @Override
            public void free(MemorySegment blob) {
                Arena arena;
                synchronized (blobs) {
                    arena = blobs.remove(blob);
                }
                if (arena == null)
                    throw new IllegalArgumentException("blob not allocated by this store");
                used -= blob.byteSize();
                arena.close();
            }

            @Override
            public long usedBytes() {
                return used;
            }

            @Override
            public void close() {
                used = 0;
                backstop.clean(); // at-most-once: frees every remaining blob now, not at GC
            }
        }
        return new Store();
    }
}
