package com.qxotic.jinfer.models.gptoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/**
 * The expert pre-slicing contract: a stacked {@code [experts, rows, cols/elementsPerBlock]} weight
 * must slice zero-copy along the expert axis and flatten each expert into a 2D row-major view, and
 * must reject malformed shapes / non-contiguous layouts before reaching the hot loop.
 */
final class GptOssSliceExpertsTest {

    @Test
    void slicesAndFlattensAThreeDimensionalStackedWeight() {
        try (Arena arena = Arena.ofConfined()) {
            int experts = 2, rows = 4, cols = 8;
            MemorySegment seg = arena.allocate(4L * experts * rows * cols, 64);
            for (int e = 0; e < experts; e++)
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        seg.set(
                                ValueLayout.JAVA_FLOAT_UNALIGNED,
                                4L * (e * rows * cols + r * cols + c),
                                e * 100f + r * 10f + c);

            MemoryView<MemorySegment> stacked =
                    Views.wrap(seg, DataType.FP32, Shape.flat(experts, rows, cols));
            MemoryView<MemorySegment>[] slices = GptOss.sliceExperts(stacked, experts);

            assertEquals(experts, slices.length);
            for (int e = 0; e < experts; e++) {
                assertEquals(Shape.flat(rows, cols), slices[e].shape());
                assertEquals(
                        Views.byteOffset(stacked, (long) e * rows * cols),
                        Views.byteOffset(slices[e], 0),
                        "expert " + e + " base offset");
                for (int r = 0; r < rows; r++)
                    for (int c = 0; c < cols; c++)
                        assertEquals(
                                e * 100f + r * 10f + c,
                                Views.getFloat(slices[e], (long) r * cols + c, "expert " + e));
            }
        }
    }

    @Test
    void rejectsWrongExpertAxis() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(4L * 3 * 4 * 8, 64);
            MemoryView<MemorySegment> stacked = Views.wrap(seg, DataType.FP32, Shape.flat(3, 4, 8));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GptOss.sliceExperts(stacked, 2),
                    "expert axis 3 != expertCount 2");
        }
    }

    @Test
    void rejectsNonThreeDimensionalStackedWeight() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment seg = arena.allocate(4L * 4 * 8, 64);
            MemoryView<MemorySegment> flat = Views.wrap(seg, DataType.FP32, Shape.flat(4, 8));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> GptOss.sliceExperts(flat, 2),
                    "a 2D stacked weight must not pass the 3D gate");
        }
    }
}
