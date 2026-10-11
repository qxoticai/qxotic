package com.qxotic.jinfer.models.gemmaembedding2;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;
import org.junit.jupiter.api.Test;

class GemmAttentionTest {
    private static final int HEADS = 4, KV_HEADS = 2, HEAD = 32;
    private static final int Q = HEADS * HEAD, KV = KV_HEADS * HEAD;

    private final MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(Arena.ofAuto());

    @Test
    void wholeInputTakesTheGemmPath() {
        check(new int[] {1300}, 0); // two query blocks
    }

    @Test
    void packedSegmentsWithAWindowMixBothPaths() {
        check(new int[] {5, 1200, 1, 40, 600}, 500); // long ones as GEMMs, the rest direct
    }

    @Test
    void shortInputTakesTheDirectPath() {
        check(new int[] {GemmAttention.DIRECT_KEYS}, 0);
    }

    @Test
    void shortPackedSegmentsTakeTheDirectPath() {
        check(new int[] {5, 60, 1, 40, 300, 7}, 37);
    }

    /**
     * Packed {@code lengths}, each query attending its own segment within {@code radius} (0: all of
     * it), against softmax attention in double.
     */
    private void check(int[] lengths, int radius) {
        int rows = Arrays.stream(lengths).sum();
        int[] first = new int[rows], end = new int[rows];
        for (int g = 0, at = 0; g < lengths.length; at += lengths[g++])
            for (int i = at; i < at + lengths[g]; i++) {
                first[i] = radius > 0 ? Math.max(at, i - radius) : at;
                end[i] = radius > 0 ? Math.min(at + lengths[g], i + radius + 1) : at + lengths[g];
            }
        float[] q = random(rows * Q, 1), k = random(rows * KV, 2), v = random(rows * KV, 3);
        MemoryView<MemorySegment> out = Views.allocateF32(memory, rows, Q);
        GemmAttention.prefill(
                Views.fromFloatArray(memory, q),
                Q,
                Views.fromFloatArray(memory, k),
                Views.fromFloatArray(memory, v),
                KV,
                out,
                rows,
                HEADS,
                KV_HEADS,
                HEAD,
                first,
                end,
                GemmAttention.Scratch.allocate(memory, rows, KV, Q));
        for (int i = 0; i < rows; i++)
            for (int h = 0; h < HEADS; h++) {
                int kvHead = h / (HEADS / KV_HEADS) * HEAD;
                double[] p = new double[end[i] - first[i]];
                double max = Double.NEGATIVE_INFINITY, sum = 0;
                for (int j = 0; j < p.length; j++) {
                    for (int d = 0; d < HEAD; d++)
                        p[j] +=
                                (double) q[i * Q + h * HEAD + d]
                                        * k[(first[i] + j) * KV + kvHead + d];
                    max = Math.max(max, p[j]);
                }
                for (int j = 0; j < p.length; j++) sum += p[j] = Math.exp(p[j] - max);
                for (int d = 0; d < HEAD; d++) {
                    double expected = 0;
                    for (int j = 0; j < p.length; j++)
                        expected += p[j] * v[(first[i] + j) * KV + kvHead + d];
                    assertEquals(
                            expected / sum,
                            Views.getFloat(out, (long) i * Q + h * HEAD + d, "out"),
                            1e-5,
                            "query " + i + " head " + h + " dim " + d);
                }
            }
    }

    private static float[] random(int n, long seed) {
        Random random = new Random(seed);
        float[] values = new float[n];
        for (int i = 0; i < n; i++) values[i] = random.nextFloat() - 0.5f;
        return values;
    }
}
