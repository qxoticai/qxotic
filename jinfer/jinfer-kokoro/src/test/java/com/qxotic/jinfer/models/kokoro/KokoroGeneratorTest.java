package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocator;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.List;
import org.junit.jupiter.api.Test;

final class KokoroGeneratorTest {

    @Test
    void generatorResidualRunsThreeUnscaledAdds() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryAllocator<MemorySegment> allocator = MemoryAllocators.ofArena(arena);
            KokoroLayers.Linear affine =
                    new KokoroLayers.Linear(
                            floats(allocator, new float[2 * 128], 2, 128),
                            floats(allocator, new float[2], 2),
                            128,
                            2);
            KokoroLayers.AdaIN adain = new KokoroLayers.AdaIN(affine, 1);
            KokoroLayers.Snake snake =
                    new KokoroLayers.Snake(floats(allocator, new float[] {1}, 1), 1);
            KokoroLayers.Conv1d first =
                    new KokoroLayers.Conv1d(new float[] {0, 0, 0}, null, 3, 1, 1);
            KokoroLayers.Conv1d second =
                    new KokoroLayers.Conv1d(
                            new float[] {0, 0, 0}, floats(allocator, new float[] {1}, 1), 3, 1, 1);
            KokoroGenerator.Step step =
                    new KokoroGenerator.Step(adain, snake, first, adain, snake, second, 1);
            KokoroGenerator.AdaINResBlock1 block =
                    new KokoroGenerator.AdaINResBlock1(List.of(step, step, step), 1);

            MemoryView<MemorySegment> output =
                    block.forward(
                            floats(allocator, new float[] {2, 4}, 1, 2),
                            2,
                            floats(allocator, new float[128], 1, 128),
                            allocator);

            assertArrayEquals(new float[] {5, 7}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void secondUpsampleGetsOneReflectedSampleOnTheLeft() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryAllocator<MemorySegment> allocator = MemoryAllocators.ofArena(arena);

            MemoryView<MemorySegment> output =
                    KokoroGenerator.reflectionPadLeft(
                            floats(allocator, new float[] {1, 2, 3, 10, 20, 30}, 2, 3),
                            2,
                            3,
                            allocator);

            assertArrayEquals(
                    new float[] {2, 1, 2, 3, 20, 10, 20, 30}, Views.toFloatArray(output, "output"));
        }
    }

    @Test
    void outputUsesExpForMagnitudeAndSinForPhase() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryAllocator<MemorySegment> allocator = MemoryAllocators.ofArena(arena);
            float[] values = new float[44];
            values[0] = (float) Math.log(2);
            values[1] = (float) Math.log(3);
            values[22] = (float) (Math.PI / 2);
            values[23] = (float) (-Math.PI / 2);

            KokoroDsp.Spectrum output =
                    KokoroGenerator.outputTransform(floats(allocator, values, 22, 2), 2, null);

            assertArrayEquals(new float[] {2, 3}, output.magnitude()[0], 1e-6f);
            assertArrayEquals(new float[] {1, -1}, output.phase()[0], 1e-6f);
            assertArrayEquals(new float[] {1, 1}, output.magnitude()[10], 1e-6f);
            assertArrayEquals(new float[] {0, 0}, output.phase()[10], 1e-6f);
        }
    }

    private static MemoryView<MemorySegment> floats(
            MemoryAllocator<MemorySegment> allocator, float[] values, long... shape) {
        MemoryView<MemorySegment> view = Views.allocateF32(allocator, shape);
        Views.copyFromArray(view, 0, values, 0, values.length, "fixture");
        return view;
    }
}
