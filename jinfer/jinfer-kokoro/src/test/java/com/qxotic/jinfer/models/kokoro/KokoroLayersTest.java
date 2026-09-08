package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class KokoroLayersTest {

    @Test
    void scalarConvSupportsStrideAndPadding() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var conv = new KokoroLayers.Conv1d(new float[] {1, 0, -1}, null, 3, 1, 1);

            var output =
                    conv.forward(
                            floats(allocator, new float[] {1, 2, 3, 4}, 1, 4), 4, 2, 1, allocator);

            assertArrayEquals(new float[] {-2, -2}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void transposedConvMatchesPyTorchLengthAndTapPlacement() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var conv = new KokoroLayers.ConvTranspose1d(new float[] {1, 10, 100}, null, 3, 1, 1);

            var output =
                    conv.forward(
                            floats(allocator, new float[] {1, 2}, 1, 2), 2, 2, 1, 1, allocator);

            assertArrayEquals(new float[] {10, 102, 20, 200}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void depthwiseUpsampleUsesTheCorrectKernelEnds() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var upsample = new KokoroLayers.DepthwiseUpsample(new float[] {1, 10, 100}, null, 1);

            var output =
                    upsample.forward(floats(allocator, new float[] {1, 2}, 1, 2), 2, allocator);

            assertArrayEquals(new float[] {10, 102, 20, 200}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void adaptiveNormsUsePopulationVarianceAndOnePlusGamma() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var projection =
                    new KokoroLayers.Linear(
                            floats(allocator, new float[] {0.5f, -0.5f, 1, -1}, 4, 1),
                            floats(allocator, new float[4], 4),
                            1,
                            4);
            var style = floats(allocator, new float[] {1}, 1, 1);

            var layerNorm =
                    new KokoroLayers.AdaLayerNorm(projection, 2)
                            .forward(
                                    floats(allocator, new float[] {1, 3, 2, 6}, 2, 2),
                                    2,
                                    style,
                                    allocator);
            var instanceNorm =
                    new KokoroLayers.AdaIN(projection, 2)
                            .forward(
                                    floats(allocator, new float[] {1, 3, 2, 6}, 2, 2),
                                    2,
                                    style,
                                    allocator);

            assertArrayEquals(
                    new float[] {-0.4999925f, -0.5000025f, -0.4999981f, -0.5000006f},
                    Views.toFloatArray(layerNorm, "layer norm"),
                    1e-6f);
            assertArrayEquals(
                    new float[] {-0.4999925f, 2.4999924f, -1.4999988f, -0.50000125f},
                    Views.toFloatArray(instanceNorm, "instance norm"),
                    1e-6f);
        }
    }

    @Test
    void nearestUpsampleRepeatsTemporalSamples() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var output =
                    KokoroLayers.nearestUpsample(
                            floats(allocator, new float[] {1, 2, 3, 4}, 2, 2), 2, 2, 3, allocator);

            assertArrayEquals(
                    new float[] {1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4},
                    Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void snakeUsesPerChannelAlpha() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var snake = new KokoroLayers.Snake(floats(allocator, new float[] {1, 2}, 2), 2);

            var output =
                    snake.forward(
                            floats(
                                    allocator,
                                    new float[] {0, (float) Math.PI / 2, 0, (float) Math.PI / 4},
                                    2,
                                    2),
                            2,
                            allocator);

            assertArrayEquals(
                    new float[] {0, (float) Math.PI / 2 + 1, 0, (float) Math.PI / 4 + 0.5f},
                    Views.toFloatArray(output, "output"),
                    1e-6f);
        }
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
