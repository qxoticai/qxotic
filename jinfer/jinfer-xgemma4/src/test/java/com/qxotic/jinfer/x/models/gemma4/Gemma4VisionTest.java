package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class Gemma4VisionTest {
    @Test
    void appliesIndependentNeoxRopeToXAndYHalves() {
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> value =
                    tensor(new PanamaMemoryArena(arena), new float[] {1, 2, 3, 4, 5, 6, 7, 8});
            Gemma4Vision.rope2d(value, 0, 8, 1, 2, 100f);
            assertPair(value, 0, 2, 1, 3, 1f);
            assertPair(value, 1, 3, 2, 4, 0.1f);
            assertPair(value, 4, 6, 5, 7, 2f);
            assertPair(value, 5, 7, 6, 8, 0.2f);
        }
    }

    @Test
    void emitsVisionBlockAtomicallyAndPreservesGeometry() {
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            Gemma4Vision vision =
                    new Gemma4Vision(
                            2,
                            1,
                            4,
                            1,
                            4,
                            2,
                            1,
                            2,
                            1e-6f,
                            tensor(memory, 4, 3, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1),
                            tensor(memory, 2, 2, 4, new float[16]),
                            new Clamped(
                                    tensor(memory, 2, 4, 1, 0, 0, 0, 0, 1, 0, 0),
                                    -Float.MAX_VALUE,
                                    Float.MAX_VALUE,
                                    -Float.MAX_VALUE,
                                    Float.MAX_VALUE),
                            new Gemma4Vision.Layer[0]);
            Media.Image image =
                    new Media.Image(new float[] {0.25f, 0.5f, 0.75f, 1f, 0f, 0.5f}, 1, 2, 3);
            assertEquals(2, vision.positions(image));
            assertThrows(
                    IllegalArgumentException.class, () -> vision.embed(image, 1, ignored -> {}));
            AtomicInteger calls = new AtomicInteger();
            vision.embed(
                    image,
                    2,
                    rows -> {
                        calls.incrementAndGet();
                        assertEquals(2, rows.shape().flatAt(0));
                        assertEquals(2, rows.shape().flatAt(1));
                    });
            assertEquals(1, calls.get());
        }
    }

    private static void assertPair(
            MemoryView<MemorySegment> value,
            int firstIndex,
            int secondIndex,
            float first,
            float second,
            float angle) {
        float cosine = (float) Math.cos(angle), sine = (float) Math.sin(angle);
        assertEquals(first * cosine - second * sine, get(value, firstIndex), 1e-6f);
        assertEquals(first * sine + second * cosine, get(value, secondIndex), 1e-6f);
    }

    private static MemoryView<MemorySegment> tensor(PanamaMemoryArena arena, float... values) {
        return tensor(arena, new long[] {values.length}, values);
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

    private static float get(MemoryView<MemorySegment> value, int index) {
        return Views.getFloat(value, index, "test tensor");
    }
}
