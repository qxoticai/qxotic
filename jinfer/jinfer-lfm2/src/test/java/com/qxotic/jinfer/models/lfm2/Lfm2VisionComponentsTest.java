package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.MediaProjectorContract;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class Lfm2VisionComponentsTest {

    @Test
    void smartResizePreservesTheUpstreamBudget() {
        assertArrayEquals(
                new int[] {576, 416},
                Lfm2VisionPreprocess.smartResize(640, 480, 32, 64 * 32 * 32, 256 * 32 * 32));
        assertArrayEquals(
                new int[] {256, 256},
                Lfm2VisionPreprocess.smartResize(32, 32, 32, 64 * 32 * 32, 256 * 32 * 32));
    }

    @Test
    void plansTilesRowMajorThenThumbnail() {
        Media.Image image = image(2000, 1000);
        Lfm2VisionPreprocess.Plan plan = Lfm2VisionPreprocess.plan(image, 16, 2);

        assertTrue(plan.tiled());
        assertEquals(2, plan.rows());
        assertEquals(4, plan.columns());
        assertEquals(9, plan.parts().size());
        assertEquals(1, plan.parts().get(0).row());
        assertEquals(1, plan.parts().get(0).column());
        assertEquals(2, plan.parts().get(1).column());
        assertFalse(plan.parts().get(1).thumbnail());
        assertTrue(plan.parts().getLast().thumbnail());
        assertEquals(256, Lfm2VisionPreprocess.positions(plan.parts().getFirst(), 16, 2));
    }

    @Test
    void smallImageUsesOnlyTheOverview() {
        Lfm2VisionPreprocess.Plan plan = Lfm2VisionPreprocess.plan(image(640, 480), 16, 2);
        assertFalse(plan.tiled());
        assertEquals(1, plan.parts().size());
        assertTrue(plan.parts().getFirst().thumbnail());
    }

    @Test
    void patchAndMergeOrderingAreChannelFirstAndRowMajor() {
        Media.Image image =
                new Media.Image(new float[] {0f, 0.25f, 0.5f, 0.75f, 1f, 0.125f}, 1, 2, 3);
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> patches = Lfm2VisionPreprocess.patches(image, 1, memory);
            assertArrayEquals(
                    new float[] {-1f, -0.5f, 0f, 0.5f, 1f, -0.75f},
                    Views.toFloatArray(patches, "patches"),
                    0f);

            MemoryView<MemorySegment> rows = tensor(memory, 4, 2, 1, 2, 3, 4, 5, 6, 7, 8);
            MemoryView<MemorySegment> merged = Lfm2Vision.merge(rows, 2, 2, 2, 2, memory);
            assertArrayEquals(
                    new float[] {1, 2, 3, 4, 5, 6, 7, 8}, Views.toFloatArray(merged, "merged"), 0f);
        }
    }

    @Test
    void positionInterpolationIsIdentityAtNativeSizeAndAntialiasedWhenShrinking() {
        float[] source = {1, 10, 2, 20, 3, 30, 4, 40};
        assertArrayEquals(source, Lfm2Vision.interpolatePositions(source, 2, 2, 2, 2), 0f);
        assertArrayEquals(
                new float[] {2.5f, 25f},
                Lfm2Vision.interpolatePositions(source, 2, 2, 1, 1),
                1e-6f);
    }

    @Test
    void tinyTowerHonoursTheSharedProjectorContract() {
        // Same synthetic tower Lfm2ChatTemplateTest uses, this time actually projecting: the
        // shared contract checks positions == rows, chunk shape/expiry and determinism.
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
            MediaProjectorContract.assertContract(tinyVision(memory), image(32, 32), 1);
        }
    }

    private static Lfm2Vision tinyVision(MemoryArena<MemorySegment> arena) {
        int patchVector = 3 * 16 * 16;
        return new Lfm2Vision(
                16,
                1,
                1,
                1,
                2,
                1,
                1,
                16,
                1e-6f,
                Views.allocateF32(arena, 1, patchVector),
                Views.allocateF32(arena, 1),
                new float[16 * 16],
                one(arena),
                Views.allocateF32(arena, 1),
                null,
                null,
                new Lfm2Vision.Linear(
                        Views.allocateF32(arena, 1, 4), Views.allocateF32(arena, 1), 1, 4),
                new Lfm2Vision.Linear(
                        Views.allocateF32(arena, 1, 1), Views.allocateF32(arena, 1), 1, 1),
                new Lfm2Vision.Layer[0]);
    }

    private static MemoryView<MemorySegment> one(MemoryArena<MemorySegment> arena) {
        MemoryView<MemorySegment> value = Views.allocateF32(arena, 1);
        Views.copyFromArray(value, 0, new float[] {1}, 0, 1, "test weight");
        return value;
    }

    private static Media.Image image(int width, int height) {
        return new Media.Image(new float[width * height * 3], height, width, 3);
    }

    private static MemoryView<MemorySegment> tensor(
            MemoryArena<MemorySegment> arena, int rows, int columns, float... values) {
        MemoryView<MemorySegment> view = Views.allocateF32(arena, rows, columns);
        Views.copyFromArray(view, 0, values, 0, values.length, "test tensor");
        return view;
    }
}
