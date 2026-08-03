package com.qxotic.jinfer.cache;

import com.qxotic.jinfer.LeakWatch;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.IdentityHashMap;

/**
 * Opaque payload storage for the prompt cache: one shared arena per blob, closed DETERMINISTICALLY
 * - {@link #free} and {@link #close} return the memory to the OS immediately (the budget is exact,
 * not soft by a GC cycle). Safe because the blob lifecycle is single-threaded by contract
 * (allocated → filled once → freed; only the generation worker): nothing reads a blob after free. A
 * Cleaner backstop covers an abandoned (dropped, unclosed) store: its remaining blobs degrade to
 * GC-eventually instead of leaking - shared arenas have no Cleaner of their own.
 *
 * <p>NATIVE IMAGE: GraalVM's SharedArenaSupport is mutually exclusive with VectorAPISupport, and
 * the kernels are non-negotiable - in the image every blob uses an automatic arena instead, so
 * frees degrade to GC-eventual there (the budget accounting stays exact either way).
 */
public final class CacheStore implements AutoCloseable {

    private static final java.lang.ref.Cleaner CLEANER = java.lang.ref.Cleaner.create();

    private static void closeArena(Arena arena) {
        try {
            arena.close();
        } catch (UnsupportedOperationException ignored) {
            // an automatic arena (native image) frees at GC; accounting already dropped it
        }
    }

    private final IdentityHashMap<MemorySegment, Arena> blobs = new IdentityHashMap<>();
    private final java.lang.ref.Cleaner.Cleanable backstop = CLEANER.register(this, sweepOf(blobs));
    private final Runnable leakWatch = LeakWatch.arm(this, "in-memory CacheStore");
    private volatile long used;

    private CacheStore() {}

    public static CacheStore inMemory() {
        return new CacheStore();
    }

    // the backstop action must capture the map only - capturing the store would pin it forever
    private static Runnable sweepOf(IdentityHashMap<MemorySegment, Arena> blobs) {
        return () -> {
            synchronized (blobs) {
                blobs.values().forEach(CacheStore::closeArena);
                blobs.clear();
            }
        };
    }

    /** Allocates a zero-filled writable blob of {@code bytes}. */
    public MemorySegment allocate(long bytes) {
        Arena arena = com.qxotic.jinfer.Arenas.newShared();
        MemorySegment blob;
        try {
            blob = arena.allocate(bytes, 64);
        } catch (RuntimeException | Error e) {
            closeArena(arena); // blobs are GB-scale: a failed allocation must not leak
            throw e;
        }
        synchronized (blobs) {
            blobs.put(blob, arena);
            used += bytes; // volatile += is not atomic: mutate only under the map's lock
        }
        return blob;
    }

    /** Releases a blob back to the store. */
    public void free(MemorySegment blob) {
        Arena arena;
        synchronized (blobs) {
            arena = blobs.remove(blob);
            if (arena != null) used -= blob.byteSize();
        }
        if (arena == null) throw new IllegalArgumentException("blob not allocated by this store");
        closeArena(arena);
    }

    /** Total live bytes (for budget enforcement). May be read from handler threads. */
    public long usedBytes() {
        return used;
    }

    @Override
    public void close() {
        used = 0;
        leakWatch.run(); // disarm: this store was closed properly
        backstop.clean(); // at-most-once: frees every remaining blob now, not at GC
    }
}
