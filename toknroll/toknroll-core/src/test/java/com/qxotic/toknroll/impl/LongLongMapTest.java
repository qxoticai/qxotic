package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Test;

/**
 * Regression tests for {@link LongLongMap} correctness.
 *
 * <p>Verifies that Robin Hood displacement does not cause lookup failures due to underestimated
 * {@code maxProbe}.
 */
class LongLongMapTest {

    /**
     * Regression for maxProbe underestimation due to Robin Hood swaps.
     *
     * <p>When items are displaced by later insertions, their probe distance can exceed the maximum
     * observed during insertion. The map must scan the full table after construction to compute the
     * true maxProbe.
     */
    @Test
    void robinHoodDisplacementDoesNotBreakLookup() {
        // Seed chosen to produce a table where actual maxProbe > insertion-time maxProbe.
        Random random = new Random(42);
        int count = 1000;

        long[] keys = new long[count];
        long[] values = new long[count];
        for (int i = 0; i < count; i++) {
            keys[i] = random.nextLong();
            values[i] = random.nextLong();
        }

        LongLongMap map = new LongLongMap(keys, values);

        // Every inserted key must be retrievable.
        random = new Random(42);
        for (int i = 0; i < count; i++) {
            long key = random.nextLong();
            long expected = random.nextLong();
            assertEquals(expected, map.get(key), "Lookup failed for key " + key + " at index " + i);
        }
    }

    /** Verifies correctness with dense packing (high load factor). */
    @Test
    void densePackingLookupCorrectness() {
        Random random = new Random(123);
        int count = 500;

        long[] keys = new long[count];
        long[] values = new long[count];
        Set<Long> used = new HashSet<>();
        for (int i = 0; i < count; i++) {
            long key;
            do {
                key = random.nextLong();
            } while (key == 0L || !used.add(key));
            keys[i] = key;
            values[i] = random.nextLong();
        }

        LongLongMap map = new LongLongMap(keys, values);

        for (int i = 0; i < count; i++) {
            assertEquals(values[i], map.get(keys[i]), "Dense lookup failed at index " + i);
        }
    }

    /** Verifies correctness with TikToken-style packed int pair keys. */
    @Test
    void packedIntPairKeysLookupCorrectness() {
        Random random = new Random(456);
        int count = 5000;

        long[] keys = new long[count];
        long[] values = new long[count];
        Set<Long> used = new HashSet<>();
        for (int i = 0; i < count; i++) {
            int left = random.nextInt(50000);
            int right = random.nextInt(50000);
            long key = ((long) left << 32) | (right & 0xFFFFFFFFL);
            if (key == 0L || !used.add(key)) {
                i--;
                continue;
            }
            keys[i] = key;
            values[i] =
                    ((long) random.nextInt(50000) << 32) | (random.nextInt(50000) & 0xFFFFFFFFL);
        }

        LongLongMap map = new LongLongMap(keys, values);

        for (int i = 0; i < count; i++) {
            assertEquals(values[i], map.get(keys[i]), "Packed pair lookup failed at index " + i);
        }
    }

    /** Verifies that absent keys always return {@link IntPair#NONE}. */
    @Test
    void absentKeysReturnNone() {
        long[] keys = {1L, 2L, 3L};
        long[] values = {10L, 20L, 30L};
        LongLongMap map = new LongLongMap(keys, values);

        assertEquals(IntPair.NONE, map.get(0L));
        assertEquals(IntPair.NONE, map.get(4L));
        assertEquals(IntPair.NONE, map.get(Long.MAX_VALUE));
        assertEquals(IntPair.NONE, map.get(Long.MIN_VALUE));
    }

    /** Verifies {@code getPair} convenience overload. */
    @Test
    void getPairOverloadWorks() {
        long[] keys = new long[100];
        long[] values = new long[100];
        Map<Long, Long> reference = new HashMap<>();
        Random random = new Random(789);

        for (int i = 0; i < 100; i++) {
            int left = random.nextInt(1000);
            int right = random.nextInt(1000);
            long key = ((long) left << 32) | (right & 0xFFFFFFFFL);
            long value = random.nextLong();
            keys[i] = key;
            values[i] = value;
            reference.put(key, value);
        }

        LongLongMap map = new LongLongMap(keys, values);

        for (Map.Entry<Long, Long> e : reference.entrySet()) {
            long key = e.getKey();
            int left = (int) (key >>> 32);
            int right = (int) key;
            assertEquals(e.getValue(), map.getPair(left, right));
        }
    }
}
