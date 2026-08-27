package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.MediaProjectorContract;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
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
        int visionDim = 8, merge = 2, modelDim = 4;
        // projectorDim != visionDim on purpose: Qwen3.5's mmproj MLP runs wider than the tower
        // (4096 vs 1024), and an equal-width toy would mask a merger buffer sized at visionDim.
        int projectorDim = 16;
        int patchVector = 3;
        int projectorInput = merge * merge * visionDim;

        List<MemoryView<MemorySegment>> borrowed = new ArrayList<>();
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Qwen35Vision tower =
                    tinyTower(
                            memory,
                            zeros(memory, visionDim, patchVector),
                            zeros(memory, visionDim, patchVector),
                            zeros(memory, projectorDim, projectorInput),
                            zeros(memory, modelDim, projectorDim));

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

            // The shared contract on top of the specifics above: positions == rows, FP32
            // [rows, modelDim] chunks, arena expiry, determinism, maxChunkSize gates.
            MediaProjectorContract.assertContract(tower, image, modelDim);
        }
        assertFalse(borrowed.getFirst().memory().base().scope().isAlive());
    }

    @Test
    void fp16AndBf16PatchKernelsRunBitIdenticalToFp32() {
        // The patch embedding honors the kernel FILE dtype (im2col + gemm, no widening copy).
        // One-hot kernel rows over exactly-representable pixels keep every product and partial
        // sum exact in both dtypes, so the towers must agree bitwise - a difference would mean
        // the F16 path computed something else, not just something rounder.
        int visionDim = 8, patchVector = 3, modelDim = 4;
        int projectorDim = 16, projectorInput = 32;
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            float[] kernel = new float[visionDim * patchVector];
            for (int c = 0; c < visionDim; c++) kernel[c * patchVector + c % 3] = 1f;
            MemoryView<MemorySegment> patchF32 =
                    tensor(memory, new long[] {visionDim, patchVector}, kernel);
            MemoryView<MemorySegment> patchF16 = Views.allocateF16(memory, visionDim, patchVector);
            Convert.f32ToF16(patchF32, 0, patchF16, 0, kernel.length);
            // BF16 = the top 16 bits of F32; the one-hot/exact-value setup is exact there too
            MemorySegment bf16Seg = arena.allocate(2L * kernel.length, 64);
            for (int i = 0; i < kernel.length; i++)
                bf16Seg.set(
                        ValueLayout.JAVA_SHORT_UNALIGNED,
                        2L * i,
                        (short) (Float.floatToRawIntBits(kernel[i]) >>> 16));
            MemoryView<MemorySegment> patchBF16 =
                    Views.wrap(bf16Seg, DataType.BF16, Shape.flat(visionDim, patchVector));

            // Nonzero merger weights: with zero mm matrices the merger zeroes the tokens and the
            // patch dtype becomes invisible downstream.
            float[] mm0 = new float[projectorDim * projectorInput];
            for (int r = 0; r < projectorDim; r++)
                mm0[r * projectorInput + r % projectorInput] = 1f;
            float[] mm2 = new float[modelDim * projectorDim];
            for (int r = 0; r < modelDim; r++) mm2[r * projectorDim + r % projectorDim] = 1f;
            MemoryView<MemorySegment> mm0W =
                    tensor(memory, new long[] {projectorDim, projectorInput}, mm0);
            MemoryView<MemorySegment> mm2W =
                    tensor(memory, new long[] {modelDim, projectorDim}, mm2);

            Qwen35Vision f32Tower = tinyTower(memory, patchF32, patchF32, mm0W, mm2W);
            Qwen35Vision f16Tower = tinyTower(memory, patchF16, patchF16, mm0W, mm2W);
            Qwen35Vision bf16Tower = tinyTower(memory, patchBF16, patchBF16, mm0W, mm2W);

            // 2x2, exactly-representable channels; normalize maps them onto {-1, -0.5, 0, 0.5, 1}.
            Media.Image image =
                    new Media.Image(
                            new float[] {
                                0.25f, 0.5f, 0.75f, 1f, 0f, 0.5f,
                                0.75f, 0.25f, 1f, 0.5f, 0f, 0.25f
                            },
                            2,
                            2,
                            3);
            float[] expected = projectAll(f32Tower, image);
            assertArrayEquals(expected, projectAll(f16Tower, image), 0f);
            assertArrayEquals(expected, projectAll(bf16Tower, image), 0f);
        }
    }

    @Test
    void flashAttentionMatchesTheReferenceSoftmax() {
        // Random (seeded) weights so every attention head, the M-RoPE and the merger carry
        // signal; the two paths differ only in summation order (online vs explicit softmax).
        int visionDim = 8, headCount = 2, ffnDim = 8, merge = 2, modelDim = 4;
        int projectorDim = 16, patchVector = 3, projectorInput = merge * merge * visionDim;
        Random rnd = new Random(7);
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            Qwen35Vision.Layer layer =
                    new Qwen35Vision.Layer(
                            ones(memory, visionDim),
                            random(memory, rnd, visionDim),
                            random(memory, rnd, 3 * visionDim, visionDim),
                            random(memory, rnd, 3 * visionDim),
                            random(memory, rnd, visionDim, visionDim),
                            random(memory, rnd, visionDim),
                            ones(memory, visionDim),
                            random(memory, rnd, visionDim),
                            random(memory, rnd, ffnDim, visionDim),
                            random(memory, rnd, ffnDim),
                            random(memory, rnd, visionDim, ffnDim),
                            random(memory, rnd, visionDim));
            Qwen35Vision tower =
                    new Qwen35Vision(
                            1,
                            visionDim,
                            modelDim,
                            headCount,
                            ffnDim,
                            merge,
                            1,
                            1e-6f,
                            random(memory, rnd, visionDim, patchVector),
                            random(memory, rnd, visionDim, patchVector),
                            random(memory, rnd, visionDim),
                            random(memory, rnd, 1, visionDim),
                            ones(memory, visionDim),
                            random(memory, rnd, visionDim),
                            new Qwen35Vision.Linear(
                                    random(memory, rnd, projectorDim, projectorInput),
                                    random(memory, rnd, projectorDim),
                                    projectorDim,
                                    projectorInput),
                            new Qwen35Vision.Linear(
                                    random(memory, rnd, modelDim, projectorDim),
                                    random(memory, rnd, modelDim),
                                    modelDim,
                                    projectorDim),
                            new Qwen35Vision.Layer[] {layer, layer});
            float[] pixels = new float[4 * 4 * 3];
            for (int i = 0; i < pixels.length; i++) pixels[i] = rnd.nextFloat();
            Media.Image image = new Media.Image(pixels, 4, 4, 3);

            String property = "jinfer.qwen35.visionFlash";
            String saved = System.getProperty(property);
            float[] reference, flash;
            try {
                System.setProperty(property, "false");
                assertFalse(Qwen35Vision.flashAttention());
                reference = projectAll(tower, image);
                System.setProperty(property, "true");
                assertTrue(Qwen35Vision.flashAttention());
                flash = projectAll(tower, image);
            } finally {
                if (saved == null) System.clearProperty(property);
                else System.setProperty(property, saved);
            }
            float scale = 0f;
            for (float v : reference) scale = Math.max(scale, Math.abs(v));
            assertTrue(scale > 0.1f, "reference output carries signal: " + scale);
            assertArrayEquals(reference, flash, scale * 1e-5f);
        }
    }

    private static MemoryView<MemorySegment> random(
            MemoryArena<MemorySegment> arena, Random rnd, long... dims) {
        long n = 1;
        for (long d : dims) n *= d;
        float[] values = new float[Math.toIntExact(n)];
        for (int i = 0; i < values.length; i++) values[i] = (rnd.nextFloat() - 0.5f);
        return tensor(arena, dims, values);
    }

    private static float[] projectAll(Qwen35Vision tower, Media.Image image) {
        List<float[]> chunks = new ArrayList<>();
        tower.project(
                image,
                tower.positions(image),
                chunk ->
                        chunks.add(
                                Views.toFloatArray(
                                        Views.castToSegmentBacked(chunk, "chunk"), "chunk")));
        assertEquals(1, chunks.size(), "tiny tower emits a single chunk");
        return chunks.get(0);
    }

    private static Qwen35Vision tinyTower(
            MemoryArena<MemorySegment> memory,
            MemoryView<MemorySegment> patch0,
            MemoryView<MemorySegment> patch1,
            MemoryView<MemorySegment> mm0Weight,
            MemoryView<MemorySegment> mm2Weight) {
        int patchSize = 1, visionDim = 8, headCount = 2, ffnDim = 8, merge = 2;
        int modelDim = 4, positionSide = 1, projectorDim = 16;
        int patchVector = 3, projectorInput = merge * merge * visionDim;
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
        return new Qwen35Vision(
                patchSize,
                visionDim,
                modelDim,
                headCount,
                ffnDim,
                merge,
                positionSide,
                1e-6f,
                patch0,
                patch1,
                zeros(memory, visionDim),
                zeros(memory, positionSide * positionSide, visionDim),
                ones(memory, visionDim),
                zeros(memory, visionDim),
                new Qwen35Vision.Linear(
                        mm0Weight, zeros(memory, projectorDim), projectorDim, projectorInput),
                new Qwen35Vision.Linear(mm2Weight, zeros(memory, modelDim), modelDim, projectorDim),
                new Qwen35Vision.Layer[] {layer});
    }

    private static MemoryView<MemorySegment> ones(MemoryArena<MemorySegment> arena, long d0) {
        float[] values = new float[Math.toIntExact(d0)];
        Arrays.fill(values, 1f);
        return tensor(arena, new long[] {d0}, values);
    }

    private static MemoryView<MemorySegment> zeros(MemoryArena<MemorySegment> arena, long d0) {
        return tensor(arena, new long[] {d0}, new float[Math.toIntExact(d0)]);
    }

    private static MemoryView<MemorySegment> zeros(
            MemoryArena<MemorySegment> arena, long d0, long d1) {
        return tensor(arena, new long[] {d0, d1}, new float[Math.toIntExact(d0 * d1)]);
    }

    private static MemoryView<MemorySegment> tensor(
            MemoryArena<MemorySegment> arena, long[] shape, float[] values) {
        MemoryView<MemorySegment> view = Views.allocateF32(arena, shape);
        assertEquals(view.shape().size(), values.length);
        Views.copyFromArray(view, 0, values, 0, values.length, "test tensor");
        return view;
    }
}
