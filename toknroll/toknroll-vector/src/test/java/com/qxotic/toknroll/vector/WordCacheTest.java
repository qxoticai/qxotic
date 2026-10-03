package com.qxotic.toknroll.vector;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class WordCacheTest {

    /** A word's ids are a function of the word, so any hit can be checked. */
    private static int[] idsOf(byte[] word, int length) {
        int h = 17;
        for (int i = 0; i < length; i++) {
            h = h * 31 + word[i];
        }
        int count = 1 + (h & 3);
        int[] ids = new int[count];
        for (int k = 0; k < count; k++) {
            ids[k] = (h * (k + 1)) & 0x7FFFFFFF;
        }
        return ids;
    }

    @Test
    void hitsAreExactAcrossGrowth() {
        WordCache cache = new WordCache(1 << 20);
        Random random = new Random(42);
        int words = 1_500_000; // through every doubling and past the largest table
        byte[] text = new byte[16 + FastEncoder.PADDING];
        int[] found = new int[WordCache.MAX_IDS + 8];
        long hits = 0;
        for (int round = 0; round < 2; round++) {
            random.setSeed(42);
            for (int w = 0; w < words; w++) {
                int length = 2 + random.nextInt(WordCache.MAX_BYTES - 1);
                for (int i = 0; i < length; i++) {
                    text[i] = (byte) ('a' + random.nextInt(26));
                }
                int[] ids = idsOf(text, length);
                int count = cache.lookup(text, 0, length, found, 3);
                if (count > 0) {
                    hits++;
                    assertEquals(ids.length, count);
                    for (int k = 0; k < count; k++) {
                        assertEquals(ids[k], found[3 + k], "id " + k + " of " + w);
                    }
                    continue;
                }
                cache.insert(ids, 0, ids.length);
                // the word just inserted is found
                assertEquals(ids.length, cache.lookup(text, 0, length, found, 0));
                assertEquals(ids[0], found[0]);
            }
        }
        // the second round replays the first: with a million slots most of it is still there
        assertTrue(hits > words / 2, "hits " + hits);
    }

    @Test
    void wordsOfEveryLengthKeepTheirOwnKey() {
        WordCache cache = new WordCache(1 << 20);
        int[] found = new int[8];
        for (int length = 2; length <= WordCache.MAX_BYTES; length++) {
            byte[] text = new byte[length + FastEncoder.PADDING];
            Arrays.fill(text, 0, length, (byte) 'x');
            assertEquals(0, cache.lookup(text, 0, length, found, 0));
            cache.insert(new int[] {length}, 0, 1);
        }
        for (int length = 2; length <= WordCache.MAX_BYTES; length++) {
            byte[] text = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".getBytes(StandardCharsets.US_ASCII);
            assertEquals(1, cache.lookup(text, 0, length, found, 0));
            assertEquals(length, found[0]);
        }
    }
}
