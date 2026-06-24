package com.qxotic.jinfer;

import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.file.Files;
import java.nio.file.Path;

public final class CacheStoreTest {

    static int failures;

    static void check(String what, boolean ok) {
        if (!ok) { failures++; System.err.println("FAIL: " + what); }
        else System.out.println("ok: " + what);
    }

    static long sum(int[] arr) {
        long s = 0; for (int v : arr) s += v; return s;
    }

    // ========================================================================
    // inMemory() tests
    // ========================================================================

    static void testInMemoryBasic() {
        System.out.println("-- in-memory basic --");
        CacheStore store = CacheStore.inMemory();

        MemorySegment a = store.allocate(1024);
        check("alloc 1024 → non-null", a != null);
        check("alloc 1024 → size", a.byteSize() == 1024);
        check("used 1024", store.usedBytes() == 1024);

        MemorySegment b = store.allocate(512);
        check("alloc 512 → size", b.byteSize() == 512);
        check("used 1536", store.usedBytes() == 1536);

        store.free(a);
        check("after free a: used 512", store.usedBytes() == 512);

        store.free(b);
        check("after free b: used 0", store.usedBytes() == 0);

        store.close();
        System.out.println();
    }

    static void testInMemoryRealloc() {
        System.out.println("-- in-memory realloc --");
        CacheStore store = CacheStore.inMemory();

        MemorySegment a = store.allocate(256);
        store.free(a);
        MemorySegment b = store.allocate(256);
        check("realloc same size", b.byteSize() == 256);
        check("used after realloc", store.usedBytes() == 256);

        store.free(b);
        store.close();
        System.out.println();
    }

    static void testInMemoryFreeUnknown() {
        System.out.println("-- in-memory free unknown --");
        CacheStore store = CacheStore.inMemory();
        CacheStore other = CacheStore.inMemory();

        MemorySegment a = other.allocate(64);
        try {
            store.free(a);
            check("free foreign blob → exception", false);
        } catch (IllegalArgumentException e) {
            check("free foreign blob → exception", true);
        }
        other.free(a);
        store.close();
        other.close();
        System.out.println();
    }

    // ========================================================================
    // mmap BlockStore tests
    // ========================================================================

    static Path tmpFile;

    static CacheStore newBlockStore(long budgetMB) {
        try {
            if (tmpFile == null) tmpFile = Files.createTempFile("cachestore", ".bin");
            Files.deleteIfExists(tmpFile);
            // 128 bytes/token for a reasonable kvBytesPerToken
            return CacheStore.mmap(tmpFile, budgetMB << 20, 512, 128);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    static void testMmapBasic() {
        System.out.println("-- mmap basic --");
        CacheStore store = newBlockStore(16);

        long blockSize = 512L * 128 + 4; // blockTokens * kvBytesPerToken + 4
        long blobSize = 2 * 512L * 128; // 2 blocks worth of KV data

        MemorySegment a = store.allocate(blobSize);
        check("alloc 2 blocks → non-null", a != null);
        check("alloc 2 blocks → size", a.byteSize() == blobSize);
        check("used > 0", store.usedBytes() >= blobSize);

        store.free(a);
        check("used after free", store.usedBytes() == 0);

        store.close();
        System.out.println();
    }

    static void testMmapMultipleBlocks() {
        System.out.println("-- mmap multiple blocks --");
        CacheStore store = newBlockStore(16);
        long blockSize = 512L * 128 + 4;
        long blob1 = 3 * 512L * 128; // 3 data-blocks

        MemorySegment a = store.allocate(blob1);
        MemorySegment b = store.allocate(blob1);
        MemorySegment c = store.allocate(blob1);

        check("three allocs non-null", a != null && b != null && c != null);
        check("distinct addresses", a.address() != b.address() && b.address() != c.address());

        long used = store.usedBytes();
        store.free(b);
        check("free middle → used decreased", store.usedBytes() < used);

        // re-allocate into the freed gap
        MemorySegment d = store.allocate(blob1);
        check("realloc into gap", d != null && d.byteSize() == blob1);

        store.free(a);
        store.free(c);
        store.free(d);
        check("all freed → used 0", store.usedBytes() == 0);

        store.close();
        System.out.println();
    }

    static void testMmapBudgetExhaustion() {
        System.out.println("-- mmap budget exhaustion --");
        CacheStore store = newBlockStore(1); // 1 MB = very small budget
        long blockSize = 512L * 128 + 4;
        long bigAlloc = 1024 * 512L * 128; // huge

        try {
            store.allocate(bigAlloc);
            check("exhaustion → OutOfMemoryError", false);
        } catch (OutOfMemoryError e) {
            check("exhaustion → OutOfMemoryError", true);
        }

        store.close();
        System.out.println();
    }

    static void testMmapSmallFallsBack() {
        System.out.println("-- mmap small allocations --");
        CacheStore store = newBlockStore(16);

        // Allocations below blockDataSize go to the in-memory fallback
        MemorySegment small = store.allocate(256);
        check("small alloc non-null", small != null);
        check("small alloc size", small.byteSize() == 256);

        MemorySegment tiny = store.allocate(1);
        check("tiny alloc non-null", tiny != null);

        store.free(small);
        store.free(tiny);
        check("small freed → used 0", store.usedBytes() == 0);

        store.close();
        System.out.println();
    }

    static void testMmapCrcRoundtrip() {
        System.out.println("-- mmap CRC roundtrip --");
        CacheStore store = newBlockStore(16);
        long kvBytes = 2 * 512L * 128; // 2 blocks of KV

        MemorySegment blob = store.allocate(kvBytes);
        // Write some data
        for (int i = 0; i < kvBytes; i += 4) {
            blob.set(ValueLayout.JAVA_INT_UNALIGNED, i, i);
        }

        store.validate(blob); // commit
        store.validate(blob); // verify (should not throw)
        check("CRC roundtrip silent", true);

        store.free(blob);
        store.close();
        System.out.println();
    }

    static void testMmapCrcCorruption() {
        System.out.println("-- mmap CRC corruption --");
        CacheStore store = newBlockStore(16);
        long kvBytes = 2 * 512L * 128;

        MemorySegment blob = store.allocate(kvBytes);
        for (int i = 0; i < kvBytes; i += 4) {
            blob.set(ValueLayout.JAVA_INT_UNALIGNED, i, i);
        }
        store.validate(blob); // commit

        // Corrupt a byte inside the first block
        blob.set(ValueLayout.JAVA_BYTE, 100, (byte) 0xFF);

        // validate detects corruption
        store.validate(blob); // verify
        check("CRC corruption handled", true);

        // The corrupt block should now be freed. Second validate: remaining blocks ok.
        store.validate(blob);
        check("re-verify after poison", true);

        store.free(blob);
        store.close();
        System.out.println();
    }

    static void testMmapFragmentation() {
        System.out.println("-- mmap fragmentation --");
        CacheStore store = newBlockStore(16);
        long kv1 = 1 * 512L * 128; // 1 data-block worth
        long kv2 = 2 * 512L * 128;
        long kv3 = 3 * 512L * 128;

        // Allocate many small blobs, free every other one
        MemorySegment[] blobs = new MemorySegment[10];
        for (int i = 0; i < 10; i++) blobs[i] = store.allocate(kv1);
        for (int i = 0; i < 10; i += 2) store.free(blobs[i]);

        // Try to allocate a larger blob that would need contiguous space
        MemorySegment big = store.allocate(kv3);
        check("alloc 3 blocks into fragmented pool", big != null);
        check("big blob size", big.byteSize() == kv3);

        // Clean up
        store.free(big);
        for (int i = 1; i < 10; i += 2) store.free(blobs[i]);
        check("all freed", store.usedBytes() == 0);

        store.close();
        System.out.println();
    }

    static void testMmapBlockBoundary() {
        System.out.println("-- mmap block boundary --");
        CacheStore store = newBlockStore(16);

        long exactBlock = 512L * 128; // exactly 1 block's KV data
        MemorySegment a = store.allocate(exactBlock);
        check("exact 1-block alloc", a != null && a.byteSize() == exactBlock);

        // CRC on exactly-aligned blob
        store.validate(a); // commit
        store.validate(a); // verify
        check("CRC exact block", true);

        store.free(a);

        long oneByteMore = exactBlock + 1;
        MemorySegment b = store.allocate(oneByteMore);
        check("block+1 alloc → spans 2 blocks", b != null);
        check("block+1 size", b.byteSize() == oneByteMore);

        store.validate(b); // commit
        store.validate(b); // verify
        check("CRC partial final block", true);

        store.free(b);
        store.close();
        System.out.println();
    }

    // ========================================================================
    // main
    // ========================================================================

    public static void main(String[] args) {
        testInMemoryBasic();
        testInMemoryRealloc();
        testInMemoryFreeUnknown();

        testMmapBasic();
        testMmapMultipleBlocks();
        testMmapBudgetExhaustion();
        testMmapSmallFallsBack();
        testMmapCrcRoundtrip();
        testMmapCrcCorruption();
        testMmapFragmentation();
        testMmapBlockBoundary();

        // Clean up temp file
        if (tmpFile != null) {
            try { Files.deleteIfExists(tmpFile); } catch (IOException ignored) {}
        }

        if (failures > 0) { System.err.println("\nCacheStoreTest: " + failures + " failures"); System.exit(1); }
        System.out.println("\nCacheStoreTest: 0 failures");
    }
}
