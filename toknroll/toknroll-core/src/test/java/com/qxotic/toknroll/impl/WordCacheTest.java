package com.qxotic.toknroll.impl;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.toknroll.IntSequence;
import java.nio.charset.StandardCharsets;
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
        WordCache cache = new WordCache();
        Random random = new Random(42);
        int words = 1_500_000; // through every doubling and past the largest table
        byte[] text = new byte[16 + TiktokenSplitter.PADDING];
        IntSequence.Builder out = IntSequence.newBuilder(8);
        int[] scratch = new int[WordCache.MAX_IDS + 8];
        long hits = 0;
        for (int round = 0; round < 2; round++) {
            random.setSeed(42);
            for (int w = 0; w < words; w++) {
                int length = 2 + random.nextInt(WordCache.MAX_BYTES - 1);
                for (int i = 0; i < length; i++) {
                    text[i] = (byte) ('a' + random.nextInt(26));
                }
                int[] ids = idsOf(text, length);
                int found = cache.lookup(text, 0, length, scratch, 3);
                if (found > 0) {
                    hits++;
                    assertEquals(ids.length, found);
                    for (int k = 0; k < found; k++) {
                        assertEquals(ids[k], scratch[3 + k], "id " + k + " of " + w);
                    }
                    continue;
                }
                out = IntSequence.newBuilder(8);
                for (int id : ids) {
                    out.add(id);
                }
                cache.insert(out, 0, ids.length);
                // the word just inserted is found, through both lookups
                assertEquals(ids.length, cache.lookup(text, 0, length, scratch, 0));
                IntSequence.Builder again = IntSequence.newBuilder(8);
                assertEquals(ids.length, cache.lookup(text, 0, length, again));
                assertEquals(ids[0], again.asSequenceView().intAt(0));
            }
        }
        // the second round replays the first: with a million slots most of it is still there
        assertTrue(hits > words / 2, "hits " + hits);
    }

    @Test
    void wordsOfEveryLengthKeepTheirOwnKey() {
        WordCache cache = new WordCache();
        int[] scratch = new int[8];
        for (int length = 2; length <= WordCache.MAX_BYTES; length++) {
            byte[] text = new byte[length + TiktokenSplitter.PADDING];
            java.util.Arrays.fill(text, (byte) 'x');
            IntSequence.Builder out = IntSequence.newBuilder(1).add(length);
            assertEquals(0, cache.lookup(text, 0, length, scratch, 0));
            cache.insert(out, 0, 1);
        }
        for (int length = 2; length <= WordCache.MAX_BYTES; length++) {
            byte[] text = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx".getBytes(StandardCharsets.US_ASCII);
            assertEquals(1, cache.lookup(text, 0, length, scratch, 0));
            assertEquals(length, scratch[0]);
        }
    }
}
