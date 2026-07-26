package com.qxotic.jinfer.cache;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.Set;

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
     * Default backend: one automatic arena per blob. GC owns the memory - {@link #free} drops the
     * reference and the Cleaner reclaims after collection, and dropping the whole store (even
     * unclosed) releases everything with it, so an abandoned cache can never leak natively. The
     * budget is therefore soft by one GC cycle: an evicted blob's memory lingers until collected.
     */
    static CacheStore inMemory() {
        return new CacheStore() {
            private final Set<MemorySegment> blobs =
                    Collections.newSetFromMap(new IdentityHashMap<>());
            private volatile long used;

            @Override
            public MemorySegment allocate(long bytes) {
                MemorySegment blob = Arena.ofAuto().allocate(bytes, 64);
                blobs.add(blob);
                used += bytes;
                return blob;
            }

            @Override
            public void free(MemorySegment blob) {
                if (!blobs.remove(blob))
                    throw new IllegalArgumentException("blob not allocated by this store");
                used -= blob.byteSize();
            }

            @Override
            public long usedBytes() {
                return used;
            }

            @Override
            public void close() {
                blobs.clear();
                used = 0;
            }
        };
    }
}
