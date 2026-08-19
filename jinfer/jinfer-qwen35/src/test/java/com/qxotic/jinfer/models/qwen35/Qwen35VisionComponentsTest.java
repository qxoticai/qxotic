package com.qxotic.jinfer.models.qwen35;

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

class Qwen35VisionComponentsTest {

    @Test
    void smartResizeHonorsTheLlamaQwen3vlBudget() {
        // 640x480 is already a multiple of align 32 and inside [8, 4096] merged tokens.
        assertArrayEquals(
                new int[] {640, 480}, Qwen35VisionPreprocess.smartResize(640, 480, 32, 8, 4096));
        // Tiny images upscale to the minimum pixel area.
        assertArrayEquals(
                new int[] {96, 96}, Qwen35VisionPreprocess.smartResize(32, 32, 32, 8, 4096));
        // Huge images downscale to the maximum pixel area (4096 * 32^2).
        assertArrayEquals(
                new int[] {2048, 2048},
                Qwen35VisionPreprocess.smartResize(4096, 4096, 32, 8, 4096));
    }

    @Test
    void smartResizeRejectsNonsense() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Qwen35VisionPreprocess.smartResize(0, 480, 32, 8, 4096));
        assertThrows(
                IllegalArgumentException.class,
                () -> Qwen35VisionPreprocess.smartResize(640, 480, 0, 8, 4096));
        assertThrows(
                IllegalArgumentException.class,
                () -> Qwen35VisionPreprocess.smartResize(640, 480, 32, 4096, 8));
    }

    @Test
    void normalizeIsChannelFirstAndCentersPixelSpace() {
        // One RGB pixel (1x1) upsampled to 2x2: every output pixel is that same pixel, but the
        // layout is CHW and values are shifted from [0,1] to [-1,1].
        Media.Image image = new Media.Image(new float[] {0.25f, 0.5f, 0.75f}, 1, 1, 3);
        float[] out = Qwen35VisionPreprocess.normalize(image, 2, 2);

        assertEquals(12, out.length); // 3 channels * 2 * 2
        float[] expected = {
            -0.5f, -0.5f, -0.5f, -0.5f, // R plane
            0f, 0f, 0f, 0f, // G plane
            0.5f, 0.5f, 0.5f, 0.5f // B plane
        };
        assertArrayEquals(expected, out, 0f);
    }

    @Test
    void normalizeCenterFitsAndPadsBlackInsteadOfStretching() {
        // 1x2 image -> 4x4 target. Scale-to-fit gives 2x4 content centered horizontally, so the
        // left and right columns are black pad (-1 after normalization).
        Media.Image image = new Media.Image(new float[] {0.25f, 0.5f, 0.75f, 1f, 0f, 0f}, 2, 1, 3);
        float[] out = Qwen35VisionPreprocess.normalize(image, 4, 4);
        int plane = 16;
        for (int y = 0; y < 4; y++) {
            for (int c = 0; c < 3; c++) {
                assertEquals(-1f, out[c * plane + y * 4 + 0], 0f, "left pad column");
                assertEquals(-1f, out[c * plane + y * 4 + 3], 0f, "right pad column");
            }
        }
        // Content column x=1 at y=0 samples the first source pixel only.
        assertEquals(-0.5f, out[1], 0f);
        assertEquals(0f, out[plane + 1], 0f);
        assertEquals(0.5f, out[2 * plane + 1], 0f);
    }

    @Test
    void positionsCountPostMergeRows() {
        Media.Image image = new Media.Image(new float[640 * 480 * 3], 480, 640, 3);
        assertEquals(300, Qwen35VisionPreprocess.positions(image, 16, 2));
    }

    @Test
    void tinyTowerRunsEndToEndAndExpiresSinkView() {
        int patchSize = 1, visionDim = 8, headCount = 2, ffnDim = 8, merge = 2;
        int modelDim = 4, positionSide = 1, projectorDim = 8;
        int patchVector = 3;
        int projectorInput = merge * merge * visionDim;

        List<MemoryView<MemorySegment>> borrowed = new ArrayList<>();
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            Qwen35Vision.Layer layer =
                    new Qwen35Vision.Layer(
                            ones(memory, visionDim),
                            zeros(memory, visionDim),
                            zeros(memory, 3 * visionDim, visionDim),
                            zeros(memory, 3 * visionDim),
                            zeros(memory, visionDim, visionDim),
                            zeros(memory, visionDim),
                            ones(memory, visionDim),
                            zeros(memory, visionDim),
                            zeros(memory, ffnDim, visionDim),
                            zeros(memory, ffnDim),
                            zeros(memory, visionDim, ffnDim),
                            zeros(memory, visionDim));
            Qwen35Vision tower =
                    new Qwen35Vision(
                            patchSize,
                            visionDim,
                            modelDim,
                            headCount,
                            ffnDim,
                            merge,
                            positionSide,
                            1e-6f,
                            zeros(memory, visionDim, patchVector),
                            zeros(memory, visionDim, patchVector),
                            zeros(memory, visionDim),
                            zeros(memory, positionSide * positionSide, visionDim),
                            ones(memory, visionDim),
                            zeros(memory, visionDim),
                            new Qwen35Vision.Linear(
                                    zeros(memory, projectorDim, projectorInput),
                                    zeros(memory, projectorDim),
                                    projectorDim,
                                    projectorInput),
                            new Qwen35Vision.Linear(
                                    zeros(memory, modelDim, projectorDim),
                                    zeros(memory, modelDim),
                                    modelDim,
                                    projectorDim),
                            new Qwen35Vision.Layer[] {layer});

            Media.Image image = new Media.Image(new float[2 * 2 * 3], 2, 2, 3);
            // A 2x2 image is upscaled by the 8-token minimum to a 6x6 patch grid (merge 2 -> 9
            // rows).
            assertEquals(9, tower.positions(image));
            assertTrue(tower.planId().contains("qwen3vl"));

            int[] rows = {0};
            tower.project(
                    image,
                    9,
                    chunk -> {
                        MemoryView<MemorySegment> view = Views.castToSegmentBacked(chunk, "chunk");
                        assertTrue(view.memory().base().scope().isAlive());
                        assertEquals(9, view.shape().flatAt(0));
                        assertEquals(modelDim, view.shape().flatAt(1));
                        float[] values = Views.toFloatArray(view, "chunk");
                        for (float v : values) assertEquals(0f, v, 0f);
                        rows[0] += Math.toIntExact(view.shape().flatAt(0));
                        borrowed.add(view);
                    });
            assertEquals(9, rows[0]);
        }
        assertFalse(borrowed.getFirst().memory().base().scope().isAlive());
    }

    private static MemoryView<MemorySegment> ones(PanamaMemoryArena arena, long d0) {
        float[] values = new float[Math.toIntExact(d0)];
        java.util.Arrays.fill(values, 1f);
        return tensor(arena, new long[] {d0}, values);
    }

    private static MemoryView<MemorySegment> zeros(PanamaMemoryArena arena, long d0) {
        return tensor(arena, new long[] {d0}, new float[Math.toIntExact(d0)]);
    }

    private static MemoryView<MemorySegment> zeros(PanamaMemoryArena arena, long d0, long d1) {
        return tensor(arena, new long[] {d0, d1}, new float[Math.toIntExact(d0 * d1)]);
    }

    private static MemoryView<MemorySegment> tensor(
            PanamaMemoryArena arena, long[] shape, float[] values) {
        MemoryView<MemorySegment> view = Views.allocateF32(arena, shape);
        assertEquals(view.shape().size(), values.length);
        Views.copyFromArray(view, 0, values, 0, values.length, "test tensor");
        return view;
    }
}
