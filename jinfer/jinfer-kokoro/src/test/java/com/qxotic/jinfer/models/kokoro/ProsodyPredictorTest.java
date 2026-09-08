package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

final class ProsodyPredictorTest {

    @Test
    void buildsCompactRepeatAlignmentAndGathersDurationRows() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            int[] alignment = ProsodyPredictor.repeatAlignment(new int[] {2, 1, 3}, null);

            var gathered =
                    ProsodyPredictor.gatherRows(
                            floats(allocator, new float[] {1, 10, 2, 20, 3, 30}, 3, 2),
                            alignment,
                            2,
                            allocator);

            assertArrayEquals(new int[] {0, 0, 1, 2, 2, 2}, alignment);
            assertArrayEquals(
                    new float[] {1, 10, 1, 10, 2, 20, 3, 30, 3, 30, 3, 30},
                    Views.toFloatArray(gathered, "gathered"));
        }
    }

    @Test
    void appendsPredictorStyleToEveryTimeStep() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);

            var result =
                    ProsodyPredictor.appendStyle(
                            floats(allocator, new float[] {1, 2, 3, 4}, 2, 2),
                            floats(allocator, new float[] {8, 9}, 1, 2),
                            2,
                            2,
                            2,
                            allocator);

            assertArrayEquals(
                    new float[] {1, 2, 8, 9, 3, 4, 8, 9}, Views.toFloatArray(result, "styled"));
        }
    }

    @Test
    void upsampledResidualUsesNearestBiasFreeShortcutAndUnitVarianceScale() {
        try (Arena arena = Arena.ofConfined()) {
            var allocator = MemoryAllocators.ofArena(arena);
            var block =
                    new ProsodyPredictor.AdainResBlk1d(
                            adain(allocator, 1),
                            adain(allocator, 2),
                            new KokoroLayers.DepthwiseUpsample(new float[] {0, 1, 0}, null, 1),
                            new KokoroLayers.Conv1d(
                                    new float[6], floats(allocator, new float[2], 2), 3, 1, 2),
                            new KokoroLayers.Conv1d(
                                    new float[12], floats(allocator, new float[2], 2), 3, 2, 2),
                            new KokoroLayers.Conv1d(new float[] {2, -1}, null, 1, 1, 2));

            var output =
                    block.forward(
                            floats(allocator, new float[] {1, 3}, 1, 2),
                            2,
                            floats(allocator, new float[] {0}, 1, 1),
                            allocator);

            float scale = (float) (1 / Math.sqrt(2));
            assertArrayEquals(
                    new float[] {
                        2 * scale,
                        2 * scale,
                        6 * scale,
                        6 * scale,
                        -scale,
                        -scale,
                        -3 * scale,
                        -3 * scale
                    },
                    Views.toFloatArray(output, "residual"),
                    1e-6f);
        }
    }

    private static KokoroLayers.AdaIN adain(
            MemoryAllocator<MemorySegment> allocator, int channels) {
        return new KokoroLayers.AdaIN(
                new KokoroLayers.Linear(
                        floats(allocator, new float[2 * channels], 2L * channels, 1),
                        floats(allocator, new float[2 * channels], 2L * channels),
                        1,
                        2 * channels),
                channels);
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
