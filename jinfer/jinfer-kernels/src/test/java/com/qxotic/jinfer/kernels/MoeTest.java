package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class MoeTest {

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
