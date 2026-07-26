package com.qxotic.jinfer.cache;

import java.lang.foreign.MemorySegment;

public final class CacheStoreTest {

    static int failures;

    static void check(String what, boolean ok) {
        if (!ok) {
            failures++;
            System.err.println("FAIL: " + what);
        } else System.out.println("ok: " + what);
    }

    static long sum(int[] arr) {
        long s = 0;
        for (int v : arr) s += v;
        return s;
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
}
