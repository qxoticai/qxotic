package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class KokoroOpsTest {

    @Test
    void matchesPyTorchLstmGateOrderInBothDirections() {
        try (Arena arena = Arena.ofShared()) {
            var allocator = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> input = floats(allocator, new float[] {0.5f, -0.25f}, 2, 1);
            MemoryView<MemorySegment> weightInput =
                    floats(allocator, new float[] {1, 2, 3, 4}, 4, 1);
            MemoryView<MemorySegment> weightHidden =
                    floats(allocator, new float[] {0.1f, 0.2f, 0.3f, 0.4f}, 4, 1);
            MemoryView<MemorySegment> biasInput =
                    floats(allocator, new float[] {0.01f, 0.02f, 0.03f, 0.04f}, 4);
            MemoryView<MemorySegment> biasHidden =
                    floats(allocator, new float[] {-0.01f, 0.01f, -0.02f, 0.02f}, 4);
            MemoryView<MemorySegment> forward = Views.allocateF32(allocator, 2, 1);
            MemoryView<MemorySegment> reverse = Views.allocateF32(allocator, 2, 1);
            MemoryView<MemorySegment> bidirectional = Views.allocateF32(allocator, 2, 2);

            KokoroOps.lstm(
                    input,
                    weightInput,
                    weightHidden,
                    biasInput,
                    biasHidden,
                    false,
                    forward,
                    allocator);
            KokoroOps.lstm(
                    input,
                    weightInput,
                    weightHidden,
                    biasInput,
                    biasHidden,
                    true,
                    reverse,
                    allocator);
            var weights =
                    new KokoroOps.LstmWeights(weightInput, weightHidden, biasInput, biasHidden);
            KokoroOps.bidirectionalLstm(input, weights, weights, bidirectional, allocator);

            assertArrayEquals(
                    new float[] {0.4535287f, -0.004156519f},
                    Views.toFloatArray(forward, "forward"),
                    1e-6f);
            assertArrayEquals(
                    new float[] {0.30372858f, -0.07547594f},
                    Views.toFloatArray(reverse, "reverse"),
                    1e-6f);
            assertArrayEquals(
                    new float[] {0.4535287f, 0.30372858f, -0.004156519f, -0.07547594f},
                    Views.toFloatArray(bidirectional, "bidirectional"),
                    1e-6f);
        }
    }

    @Test
    void resolvesDurationsWithRoundToEvenAndSpeedScaling() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            float logitForHalf = (float) -Math.log(3);
            var logits =
                    floats(allocator, new float[] {0, 0, 0, 0, logitForHalf, logitForHalf}, 2, 3);
            assertArrayEquals(new int[] {2, 1}, KokoroOps.durations(logits, 2, 3, 1, allocator));
            assertArrayEquals(
                    new int[] {1},
                    KokoroOps.durations(
                            floats(allocator, new float[] {0, 0, 0}, 1, 3), 1, 3, 2, allocator));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> KokoroOps.durations(logits, 2, 3, Double.NaN, allocator));
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            KokoroOps.durations(
                                    floats(allocator, new float[] {0, Float.NaN, 0}, 1, 3),
                                    1,
                                    3,
                                    1,
                                    allocator));
        }
    }

    private static MemoryView<MemorySegment> floats(
            com.qxotic.jota.memory.MemoryAllocator<MemorySegment> allocator,
            float[] values,
            long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
