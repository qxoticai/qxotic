package com.llama4j;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.IdentityHashMap;
import java.util.Map;

/**
 * Opaque payload storage for the prompt cache. The full blob contract: {@link #allocate}
 * returns an ordinary {@link MemorySegment} that is written exactly once, immediately, then
 * treated as IMMUTABLE and READABLE until freed — the cache never updates stored bytes in
 * place (checkpoint attach allocates a fresh blob; tombstoning frees one). The radix index
 * (tokens, topology, LRU, pins) stays in memory inside {@link PromptCache}; ALL payload
 * bytes (KV rows, conv checkpoints, bx rows) live behind this seam so alternative backends
 * can plug in without touching cache logic: zero-copy ones hand out mapped slices (mmap
 * snapshot, LMDB), copying ones (SQLite) return heap segments. Loaded read-only blobs (a
 * future L2 tier) satisfy the same contract — their write phase happened in a past life.
 *
 * <p>Single-threaded like the cache: only the generation worker allocates and frees.
 * {@link #usedBytes()} may additionally be read from HTTP handler threads for stats.
 */
interface CacheStore extends AutoCloseable {

    /** A new zero-filled, 64-byte-aligned, writable blob of exactly {@code bytes} bytes. */
    MemorySegment allocate(long bytes);

    void free(MemorySegment blob);

    /** Total bytes of live blobs; the cache budget is enforced against this. */
    long usedBytes();

    @Override
    default void close() {}

    /** Default backend: one confined arena per blob, byte-exact accounting. */
    static CacheStore inMemory() {
        return new CacheStore() {
            private final Map<MemorySegment, Arena> arenas = new IdentityHashMap<>();
            private volatile long used; // handler threads read it for /props

            @Override
            public MemorySegment allocate(long bytes) {
                Arena arena = Arena.ofConfined();
                MemorySegment blob = arena.allocate(bytes, 64);
                arenas.put(blob, arena);
                used += bytes;
                return blob;
            }

            @Override
            public void free(MemorySegment blob) {
                Arena arena = arenas.remove(blob);
                if (arena == null) {
                    throw new IllegalArgumentException("blob not allocated by this store");
                }
                used -= blob.byteSize();
                arena.close();
            }

            @Override
            public long usedBytes() {
                return used;
            }

            @Override
            public void close() {
                arenas.values().forEach(Arena::close);
                arenas.clear();
                used = 0;
            }
        };
    }
}
