package com.qxotic.toknroll.libtoknroll;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Random;
import org.junit.jupiter.api.Test;

class RunsTest {

    /** starts for the given text lengths: text i spans [starts[i], starts[i + 1]). */
    private static long[] starts(int... lengths) {
        long[] starts = new long[lengths.length + 1];
        for (int i = 0; i < lengths.length; i++) starts[i + 1] = starts[i] + lengths[i];
        return starts;
    }

    @Test
    void anEmptyBatchIsOneEmptyRun() {
        assertArrayEquals(new int[] {0, 0}, Runs.of(new long[] {0}, 0, 8));
    }

    @Test
    void fewTextsAreOneRunEach() {
        assertArrayEquals(new int[] {0, 1, 2, 3}, Runs.of(starts(5, 5, 5), 3, 8));
    }

    @Test
    void emptyTextsRideAlong() {
        int[] runs = Runs.of(starts(0, 0, 0, 0), 4, 2);
        assertEquals(0, runs[0]);
        assertEquals(4, runs[runs.length - 1]);
    }

    @Test
    void runsCoverEveryTextInOrderAndShareTheBytes() {
        Random random = new Random(7);
        for (int t = 0; t < 2000; t++) {
            int count = 1 + random.nextInt(300);
            int[] lengths = new int[count];
            for (int i = 0; i < count; i++)
                lengths[i] =
                        random.nextInt(4) == 0
                                ? 0
                                : random.nextInt(random.nextBoolean() ? 50 : 5000);
            long[] starts = starts(lengths);
            int k = 1 + random.nextInt(64);
            int[] runs = Runs.of(starts, count, k);
            assertEquals(0, runs[0]);
            assertEquals(count, runs[runs.length - 1]);
            assertTrue(runs.length - 1 <= k, "at most k runs");
            int longest = 0;
            for (int i = 0; i < count; i++) longest = Math.max(longest, lengths[i]);
            for (int r = 0; r + 1 < runs.length; r++) {
                assertTrue(runs[r] < runs[r + 1], "never empty");
                // a run holds about its share, give or take the one text that crosses a cut
                long bytes = starts[runs[r + 1]] - starts[runs[r]];
                assertTrue(
                        bytes <= starts[count] / (runs.length - 1) + 2L * longest + 1, "balanced");
            }
        }
    }
}
