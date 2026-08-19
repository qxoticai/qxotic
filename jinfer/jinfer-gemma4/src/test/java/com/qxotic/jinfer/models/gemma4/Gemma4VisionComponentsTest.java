package com.qxotic.jinfer.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class Gemma4VisionComponentsTest {

    @Test
    void preservesSmartResizeAndPatchOrdering() {
        assertArrayEquals(
                new int[] {624, 480},
                VisionPreprocess.smartResize(640, 480, 48, 48 * 48, 280 * 48 * 48));

        Media.Image image =
                new Media.Image(new float[] {0f, 0.25f, 0.5f, 0.75f, 1f, 0.125f}, 1, 2, 3);
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> patches =
                    VisionPreprocess.im2col(image, 2, 1, 1, new PanamaMemoryArena(arena));
            assertEquals(2, patches.shape().flatAt(0));
            assertEquals(3, patches.shape().flatAt(1));
            assertArrayEquals(new float[] {-1f, -0.5f, 0f, 0.5f, 1f, -0.75f}, values(patches), 0f);
        }
    }

    @Test
    void clampsWithoutMutatingInputAndChecksShapes() {
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> weight = tensor(memory, 2, 2, 1f, 0f, 0f, 1f);
            MemoryView<MemorySegment> input = tensor(memory, 1, 2, -2f, 2f);
            MemoryView<MemorySegment> output = tensor(memory, 1, 2, 0f, 0f);
            MemoryView<MemorySegment> scratch = tensor(memory, 1, 2, 0f, 0f);
            new Clamped(weight, -1f, 1f, -0.5f, 0.5f).gemm(input, 2, output, 2, 1, scratch);
            assertArrayEquals(new float[] {-2f, 2f}, values(input), 0f);
            assertArrayEquals(new float[] {-0.5f, 0.5f}, values(output), 0f);
            assertThrows(
                    IllegalArgumentException.class,
                    () ->
                            new Clamped(weight, -1f, 1f, -0.5f, 0.5f)
                                    .gemm(input, 2, tensor(memory, 1, 0f), 2, 1, scratch));
        }
    }

    @Test
    void unifiedProjectorPlansAtomicBlockAndExpiresSinkView() {
        List<MemoryView<?>> borrowed = new ArrayList<>();
        try (Arena weightsArena = Arena.ofConfined()) {
            PanamaMemoryArena weights = new PanamaMemoryArena(weightsArena);
            Gemma4VisionUnified projector =
                    new Gemma4VisionUnified(
                            1,
                            2,
                            2,
                            7,
                            1e-6f,
                            tensor(weights, 2, 3, 1f, 0f, 0f, 0f, 1f, 0f),
                            tensor(weights, 2, 0f, 0f),
                            tensor(weights, 2, 7, 2, new float[28]),
                            tensor(weights, 2, 2, 1f, 0f, 0f, 1f),
                            tensor(weights, 3, 1f, 1f, 1f),
                            tensor(weights, 3, 0f, 0f, 0f),
                            tensor(weights, 2, 1f, 1f),
                            tensor(weights, 2, 0f, 0f),
                            tensor(weights, 2, 1f, 1f),
                            tensor(weights, 2, 0f, 0f));
            Media.Image image = new Media.Image(new float[] {0.25f, 0.5f, 0.75f}, 1, 1, 3);
            assertEquals(49, projector.positions(image));
            assertThrows(
                    IllegalArgumentException.class, () -> projector.project(image, 48, v -> {}));

            int[] rows = {0};
            projector.project(
                    image,
                    49,
                    chunk -> {
                        assertTrue(
                                chunk.memory().base() instanceof MemorySegment segment
                                        && segment.scope().isAlive());
                        assertEquals(49, chunk.shape().flatAt(0));
                        assertEquals(2, chunk.shape().flatAt(1));
                        for (float value : values(Views.castToSegmentBacked(chunk, "chunk")))
                            assertTrue(Float.isFinite(value));
                        rows[0] += Math.toIntExact(chunk.shape().flatAt(0));
                        borrowed.add(chunk);
                    });
            assertEquals(49, rows[0]);
        }
        assertFalse(((MemorySegment) borrowed.getFirst().memory().base()).scope().isAlive());
    }

    private static MemoryView<MemorySegment> tensor(
            PanamaMemoryArena arena, long d0, float... values) {
        return tensor(arena, new long[] {d0}, values);
    }

    private static MemoryView<MemorySegment> tensor(
            PanamaMemoryArena arena, long d0, long d1, float... values) {
        return tensor(arena, new long[] {d0, d1}, values);
    }

    private static MemoryView<MemorySegment> tensor(
            PanamaMemoryArena arena, long d0, long d1, long d2, float[] values) {
        return tensor(arena, new long[] {d0, d1, d2}, values);
    }

    private static MemoryView<MemorySegment> tensor(
            PanamaMemoryArena arena, long[] shape, float[] values) {
        MemoryView<MemorySegment> view = Views.allocateF32(arena, shape);
        assertEquals(view.shape().size(), values.length);
        Views.copyFromArray(view, 0, values, 0, values.length, "test tensor");
        return view;
    }

    private static float[] values(MemoryView<MemorySegment> view) {
        return Views.toFloatArray(view, "test tensor");
    }
}
