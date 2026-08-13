package com.qxotic.jinfer.x.cache;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/**
 * The store's accounting and ownership laws: exact byte counts, recycling, foreign-blob refusal.
 */
public final class CacheStoreTest {

    @Test
    void allocatesFreesAndAccountsExactly() {
        CacheStore store = CacheStore.inMemory();

        MemorySegment a = store.allocate(1024);
        assertEquals(1024, a.byteSize());
        assertEquals(1024, store.usedBytes());

        MemorySegment b = store.allocate(512);
        assertEquals(512, b.byteSize());
        assertEquals(1536, store.usedBytes());

        store.free(a);
        assertEquals(512, store.usedBytes());

        store.free(b);
        assertEquals(0, store.usedBytes());

        store.close();
    }

    @Test
    void reallocatesAfterFree() {
        CacheStore store = CacheStore.inMemory();

        MemorySegment a = store.allocate(256);
        store.free(a);
        MemorySegment b = store.allocate(256);
        assertEquals(256, b.byteSize());
        assertEquals(256, store.usedBytes());

        store.free(b);
        store.close();
    }

    @Test
    void refusesABlobItDidNotAllocate() {
        CacheStore store = CacheStore.inMemory();
        CacheStore other = CacheStore.inMemory();

        MemorySegment a = other.allocate(64);
        assertThrows(IllegalArgumentException.class, () -> store.free(a));
        other.free(a);
        store.close();
        other.close();
    }
}
