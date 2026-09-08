package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class KokoroDecoderTest {

    @Test
    void gathersTimeMajorTextAndReturnsChannelMajorAsr() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            float[] text = new float[3 * 512];
            for (int row = 0; row < 3; row++)
                for (int channel = 0; channel < 512; channel++)
                    text[row * 512 + channel] = row * 1000 + channel;

            var result =
                    KokoroDecoder.gatherAsr(
                            floats(allocator, text, 3, 512), new int[] {2, 0, 2}, allocator);
            float[] expected = new float[512 * 3];
            for (int channel = 0; channel < 512; channel++) {
                expected[channel * 3] = 2000 + channel;
                expected[channel * 3 + 1] = channel;
                expected[channel * 3 + 2] = 2000 + channel;
            }

            assertEquals(512, result.shape().flatAt(0));
            assertEquals(3, result.shape().flatAt(1));
            assertArrayEquals(expected, Views.toFloatArray(result, "asr"));
        }
    }

    @Test
    void downsampleUsesStrideTwoAndSymmetricPadding() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var convolution =
                    new KokoroLayers.Conv1d(
                            new float[] {1, 10, 100},
                            floats(allocator, new float[] {5}, 1),
                            3,
                            1,
                            1);

            var result =
                    KokoroDecoder.downsampleCurve(
                            convolution, new float[] {1, 2, 3, 4, 5, 6}, 3, allocator);

            assertArrayEquals(new float[] {215, 437, 659}, Views.toFloatArray(result, "curve"));
        }
    }

    @Test
    void concatenatesDecoderInputsByChannelWithoutChangingTimeLayout() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);

            var result =
                    KokoroDecoder.concatenateConditioning(
                            floats(allocator, new float[] {1, 2, 3, 4}, 2, 2),
                            2,
                            floats(allocator, new float[] {5, 6}, 1, 2),
                            1,
                            floats(allocator, new float[] {7, 8}, 1, 2),
                            floats(allocator, new float[] {9, 10}, 1, 2),
                            2,
                            allocator);

            assertArrayEquals(
                    new float[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10},
                    Views.toFloatArray(result, "conditioned"));
        }
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
