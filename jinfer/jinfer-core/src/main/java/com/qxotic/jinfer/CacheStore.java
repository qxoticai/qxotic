package com.qxotic.jinfer;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.BitSet;
import java.util.IdentityHashMap;
import java.util.Map;
import java.util.zip.CRC32C;

/**
 * Opaque payload storage for the prompt cache — a single-implementation seam so
 * alternative backends (in-memory arena, mmap pool, future network-attached store)
 * can plug in without touching cache logic.
 *
 * <p>Blob lifecycle: allocated → filled once → maybe validated → freed. Immutable
 * after fill. Single-threaded by design (only the generation worker).
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
     * Opaque integrity hook. Callers invoke this after filling a newly allocated
     * blob (commit) and before reading a previously stored blob (verify). The
     * default does nothing; implementations that provide block-level checksums
     * handle both commit and verify internally (idempotent after first call).
     */
    default void validate(MemorySegment blob) {}

    /** Default backend: one confined arena per blob. */
    static CacheStore inMemory() {
        return new CacheStore() {
            private final Map<MemorySegment, Arena> arenas = new IdentityHashMap<>();
            private volatile long used;

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
                if (arena == null) throw new IllegalArgumentException("blob not allocated by this store");
                used -= blob.byteSize();
                arena.close();
            }

            @Override
            public long usedBytes() { return used; }

            @Override
            public void close() {
                arenas.values().forEach(Arena::close);
                arenas.clear();
                used = 0;
            }
        };
    }

    /**
     * Block-pool backend backed by a memory-mapped file. Fixed-size blocks
     * ({@code blockTokens} KV tokens each), stored contiguously. A CRC32C
     * trailer at end-of-file provides per-block integrity without fragmenting
     * the data region. Allocations under one block fall back to in-memory arenas.
     */
    static CacheStore mmap(Path file, long budgetBytes, int blockTokens, long kvBytesPerToken) throws IOException {
        return new BlockStore(file, budgetBytes, blockTokens, kvBytesPerToken);
    }
}

final class BlockStore implements CacheStore {
    private final MemorySegment pool;
    private final long blockDataSize;   // blockTokens * kvBytesPerToken
    private final int numBlocks;
    private final long crcOffset;       // start of CRC trailer within the pool
    private final BitSet usedBlocks;
    private final CacheStore small;     // for allocations under one block
    private volatile long usedBytes;

    private final CRC32C crc = new CRC32C();
    private final byte[] scratch = new byte[8192];

    BlockStore(Path file, long budgetBytes, int blockTokens, long kvBytesPerToken) throws IOException {
        this.blockDataSize = blockTokens * kvBytesPerToken;
        long dataRegion = (budgetBytes / (blockDataSize + 4)) * blockDataSize;
        this.numBlocks = Math.toIntExact(dataRegion / blockDataSize);
        this.crcOffset = dataRegion;
        this.usedBlocks = new BitSet(numBlocks);
        this.small = CacheStore.inMemory();

        long fileSize = crcOffset + (long) numBlocks * 4;
        FileChannel fc = FileChannel.open(file, StandardOpenOption.READ, StandardOpenOption.WRITE,
                StandardOpenOption.CREATE, StandardOpenOption.TRUNCATE_EXISTING);
        this.pool = fc.map(FileChannel.MapMode.READ_WRITE, 0, fileSize, Arena.global());
        fc.close();
    }

    // ---- allocation ----

    @Override
    public MemorySegment allocate(long bytes) {
        if (bytes >= blockDataSize) return allocKv(bytes);
        return small.allocate(bytes);
    }

    private MemorySegment allocKv(long bytes) {
        int needed = Math.toIntExact((bytes + blockDataSize - 1) / blockDataSize);
        int start = allocContiguous(needed);
        for (int i = start; i < start + needed; i++) usedBlocks.set(i);
        usedBytes += needed * blockDataSize;
        return pool.asSlice((long) start * blockDataSize, bytes);
    }

    private int allocContiguous(int needed) {
        for (int i = 0; i <= numBlocks - needed; ) {
            i = usedBlocks.nextClearBit(i);
            if (i > numBlocks - needed) break;
            int nextUsed = usedBlocks.nextSetBit(i);
            if (nextUsed < 0 || nextUsed >= i + needed) return i;
            i = nextUsed;
        }
        throw new OutOfMemoryError("block pool exhausted (" + numBlocks + " blocks)");
    }

    @Override
    public void free(MemorySegment blob) {
        if (inPool(blob)) {
            long off = blob.address() - pool.address();
            long bytes = blob.byteSize();
            int start = Math.toIntExact(off / blockDataSize);
            int blocks = Math.toIntExact((bytes + blockDataSize - 1) / blockDataSize);
            for (int i = start; i < start + blocks; i++) {
                if (usedBlocks.get(i)) {
                    usedBlocks.clear(i);
                    usedBytes -= blockDataSize;
                    pool.set(ValueLayout.JAVA_INT_UNALIGNED, crcOffset + (long) i * 4, 0);
                }
            }
        } else {
            small.free(blob);
        }
    }

    @Override
    public long usedBytes() { return usedBytes + small.usedBytes(); }

    @Override
    public void close() { small.close(); }

    // ---- integrity ----

    @Override
    public void validate(MemorySegment blob) {
        if (blob == null || !inPool(blob)) return;
        long off = blob.address() - pool.address();
        int start = Math.toIntExact(off / blockDataSize);
        int blocks = Math.toIntExact((blob.byteSize() + blockDataSize - 1) / blockDataSize);
        for (int b = 0; b < blocks; b++) {
            int idx = start + b;
            long crcSlot = crcOffset + (long) idx * 4;
            int stored = pool.get(ValueLayout.JAVA_INT_UNALIGNED, crcSlot);
            // CRC of the block's data range within the pool (not the blob slice)
            int actual = crc32c((long) idx * blockDataSize, blockDataSize);
            if (stored == 0) {
                pool.set(ValueLayout.JAVA_INT_UNALIGNED, crcSlot, actual);
            } else if (stored != actual) {
                System.err.println("[cache] CRC mismatch on block " + idx + " — discarding");
                for (int j = b; j < blocks; j++) {
                    if (usedBlocks.get(start + j)) {
                        usedBlocks.clear(start + j);
                        usedBytes -= blockDataSize;
                    }
                    pool.set(ValueLayout.JAVA_INT_UNALIGNED, crcOffset + (long) (start + j) * 4, 0);
                }
                return;
            }
        }
    }

    private int crc32c(long offset, long len) {
        crc.reset();
        long end = offset + len;
        for (long p = offset; p < end; ) {
            int chunk = (int) Math.min(scratch.length, end - p);
            MemorySegment.copy(pool, p, MemorySegment.ofArray(scratch), 0, chunk);
            crc.update(scratch, 0, chunk);
            p += chunk;
        }
        return (int) crc.getValue();
    }

    private boolean inPool(MemorySegment blob) {
        long addr = blob.address();
        return addr >= pool.address() && addr < pool.address() + crcOffset;
    }
}
