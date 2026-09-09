package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/** Layers that fan out over channels run on the pool, so their fixtures use shared arenas. */
final class KokoroLayersTest {

    @Test
    void scalarConvSupportsStrideAndPadding() {
        try (Arena arena = Arena.ofShared()) {
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
            var conv =
                    new KokoroLayers.ConvTranspose1d(
                            floats(allocator, new float[] {1, 10, 100}, 3, 1), null, 3, 1, 1);

            var output =
                    conv.forward(
                            floats(allocator, new float[] {1, 2}, 1, 2), 2, 2, 1, 1, allocator);

            assertArrayEquals(new float[] {10, 102, 20, 200}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void transposedConvMatchesScalarOracleAcrossKokoroShapes() {
        try (Arena arena = Arena.ofShared()) {
            var allocator = MemoryAllocators.ofArena(arena);
            Random random = new Random(42);
            int inChannels = 3, outChannels = 2, time = 7;
            float[] input = randomFloats(random, inChannels * time);
            float[] bias = randomFloats(random, outChannels);
            int[][] shapes = {{3, 2, 1, 1}, {20, 10, 5, 0}, {12, 6, 3, 0}};

            for (int[] shape : shapes) {
                int kernel = shape[0], stride = shape[1], padding = shape[2];
                int outputPadding = shape[3];
                float[] weight = randomFloats(random, outChannels * kernel * inChannels);
                var conv =
                        new KokoroLayers.ConvTranspose1d(
                                floats(allocator, weight, outChannels * kernel, inChannels),
                                floats(allocator, bias, outChannels),
                                kernel,
                                inChannels,
                                outChannels);

                var actual =
                        conv.forward(
                                floats(allocator, input, inChannels, time),
                                time,
                                stride,
                                padding,
                                outputPadding,
                                allocator);

                assertArrayEquals(
                        scalarConvTranspose(
                                input,
                                weight,
                                bias,
                                time,
                                stride,
                                padding,
                                outputPadding,
                                kernel,
                                inChannels,
                                outChannels),
                        Views.toFloatArray(actual, "output"),
                        1e-5f);
            }
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
        try (Arena arena = Arena.ofShared()) {
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
        try (Arena arena = Arena.ofShared()) {
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

    private static float[] randomFloats(Random random, int size) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = random.nextFloat() * 2 - 1;
        return values;
    }

    private static float[] scalarConvTranspose(
            float[] input,
            float[] weight,
            float[] bias,
            int time,
            int stride,
            int padding,
            int outputPadding,
            int kernel,
            int inChannels,
            int outChannels) {
        int outTime = (time - 1) * stride - 2 * padding + kernel + outputPadding;
        float[] output = new float[outChannels * outTime];
        for (int oc = 0; oc < outChannels; oc++) {
            for (int target = 0; target < outTime; target++)
                output[oc * outTime + target] = bias[oc];
            for (int k = 0; k < kernel; k++)
                for (int ic = 0; ic < inChannels; ic++)
                    for (int t = 0, target = k - padding; t < time; t++, target += stride)
                        if (target >= 0 && target < outTime)
                            output[oc * outTime + target] +=
                                    input[ic * time + t]
                                            * weight[(oc * kernel + k) * inChannels + ic];
        }
        return output;
    }
}
