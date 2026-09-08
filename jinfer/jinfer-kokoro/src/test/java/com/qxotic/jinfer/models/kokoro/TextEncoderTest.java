package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.Test;

final class TextEncoderTest {

    @Test
    void runsEmbeddingConvolutionsAndBidirectionalLstm() {
        try (Arena arena = Arena.ofShared()) {
            var allocator = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> zero = floats(allocator, new float[] {0, 0}, 2);
            MemoryView<MemorySegment> one = floats(allocator, new float[] {1, 1}, 2);
            float[] identity = {
                0, 1, 0, 0, 0, 0,
                0, 0, 0, 0, 1, 0
            };
            var convolution = new TextEncoder.Conv(identity, zero, one, zero);
            MemoryView<MemorySegment> inputWeights =
                    floats(
                            allocator,
                            new float[] {.5f, -.25f, .1f, .2f, .7f, -.4f, -.3f, .6f},
                            4,
                            2);
            MemoryView<MemorySegment> hiddenWeights =
                    floats(allocator, new float[] {.1f, -.2f, .3f, .4f}, 4, 1);
            MemoryView<MemorySegment> bias = floats(allocator, new float[4], 4);
            var lstm = new KokoroOps.LstmWeights(inputWeights, hiddenWeights, bias, bias);
            var weights =
                    new TextEncoder.Weights(
                            floats(allocator, new float[] {1, 3, 2, 0, 4, -2}, 3, 2),
                            List.of(convolution, convolution, convolution),
                            lstm,
                            lstm);

            MemoryView<MemorySegment> output =
                    TextEncoder.encode(weights, new int[] {0, 1, 2}, 2, 3, allocator);

            assertArrayEquals(
                    new float[] {
                        -.13252127f, .10728436f, .10860053f, .23320903f, .21133459f, .15534734f
                    },
                    Views.toFloatArray(output, "output"),
                    1e-6f);
            assertThrows(
                    IllegalArgumentException.class,
                    () -> TextEncoder.encode(weights, new int[] {3}, 2, 3, allocator));
        }
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
