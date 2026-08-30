package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class MoeTest {

    @Test
    void groupedRoutingUsesBiasedScoresButUnbiasedWeights() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var selection =
                    Views.fromFloatArray(
                            memory, new float[] {.9f, .8f, .7f, .6f, .5f, .4f, .99f, .01f});
            var weights =
                    Views.fromFloatArray(
                            memory, new float[] {.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f});
            int[] experts = new int[3], counts = new int[8];
            float[] probabilities = new float[3];

            Moe.selectTopKGrouped(
                    selection,
                    weights,
                    1,
                    8,
                    3,
                    4,
                    2,
                    experts,
                    probabilities,
                    counts,
                    new float[4],
                    new boolean[4]);

            assertArrayEquals(new int[] {0, 1, 2}, experts);
            assertArrayEquals(new float[] {.1f, .2f, .3f}, probabilities);
            assertArrayEquals(new int[] {1, 1, 1, 0, 0, 0, 0, 0}, counts);
        }
    }

    @Test
    void groupedRoutingKeepsBadRowsDistinctInsteadOfCrashing() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var values =
                    Views.fromFloatArray(
                            memory,
                            new float[] {
                                Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN, Float.NaN,
                                Float.NaN, Float.NaN
                            });
            int[] experts = new int[3], counts = new int[8];
            float[] probabilities = new float[3];

            Moe.selectTopKGrouped(
                    values,
                    values,
                    1,
                    8,
                    3,
                    4,
                    2,
                    experts,
                    probabilities,
                    counts,
                    new float[4],
                    new boolean[4]);

            assertArrayEquals(new int[] {0, 1, 2}, experts);
            assertArrayEquals(new int[] {1, 1, 1, 0, 0, 0, 0, 0}, counts);
        }
    }

    @Test
    void aNaNRouterRowStillRoutesToDistinctExperts() {
        // row 0 is healthy, row 1 is all NaN: the NaN must travel in the combine weight, not
        // collapse the row onto expert 0 topK times (which overflows the per-expert gather and
        // scatters the same output row from two tasks)
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float nan = Float.NaN;
            var logits =
                    Views.fromFloatArray(
                            memory, new float[] {0.1f, 0.7f, 0.2f, 0.4f, nan, nan, nan, nan});
            int[] rowTopE = new int[4];
            float[] rowTopP = new float[4];
            int[] counts = new int[4];

            Moe.selectTopK(logits, 2, 4, 2, rowTopE, rowTopP, counts);

            assertArrayEquals(new int[] {1, 3, 0, 1}, rowTopE, "distinct experts per row");
            assertArrayEquals(new int[] {1, 2, 0, 1}, counts, "no expert counted twice for a row");
            assertTrue(Float.isNaN(rowTopP[2]) && Float.isNaN(rowTopP[3]), "the NaN is kept");
        }
    }
}
