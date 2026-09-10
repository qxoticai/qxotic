package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class MoeDispatchTest {

    /**
     * Three rows, two experts, one route each, on a kernel that doubles its input: every row lands
     * on the expert its routing names, scaled by its combine weight, and the scratch sized for more
     * rows than this call uses stays inert.
     */
    @Test
    void rowsReachTheirExpertsAndComeBackWeighted() {
        int rows = 3, dim = 2, topK = 1, capacity = 4;
        int[] rowTopE = {1, 0, 1, 99}; // row 3 is beyond this call: garbage there is ignored
        float[] rowTopP = {0.5f, 1f, 2f, 99f};
        int[] counts = {1, 2};
        Moe.Routing routing = new Moe.Routing(rowTopE, rowTopP, counts, topK);
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var input = Views.fromFloatArray(memory, new float[] {1, 2, 3, 4, 5, 6, 7, 8});
            var gather = Views.allocateF32(memory, (long) capacity * dim);
            var expertOut = Views.allocateF32(memory, (long) capacity * dim);
            var out = Views.allocateF32(memory, (long) capacity * dim);
            Moe.dispatch(
                    routing,
                    rows,
                    dim,
                    input,
                    gather,
                    expertOut,
                    out,
                    null,
                    (e, n, in, o) -> {
                        Ops.fillInPlace(o, 0, n * dim, 0f);
                        Ops.saxpyInPlace(o, 0, in, 0, n * dim, 2f); // o = 2 * in
                    });
            // row 0: expert 1, weight 0.5 -> (1,2)*2*0.5; row 1: 1.0 -> (3,4)*2; row 2: 2.0
            assertArrayEquals(
                    new float[] {1, 2, 6, 8, 20, 24, 0, 0}, Views.toFloatArray(out, "out"), 0f);
        }
    }
}
