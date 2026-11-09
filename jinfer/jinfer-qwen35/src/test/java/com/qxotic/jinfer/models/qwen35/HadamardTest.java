package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

final class HadamardTest {

    // block 4 over width 8; ssm_out geometry: head 2, 2 key heads, 2 repeats
    private final Hadamard hadamard =
            new Hadamard(
                    4, Map.of(8, new int[] {1, -1, 1, 1, -1, 1, -1, -1}), new int[] {8}, 2, 2, 2);

    @Test
    void inverseUndoesForward() {
        float[] x = {0.5f, -1, 2, 3, -4, 0.25f, 7, -2};
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> v = row(arena, x), rotated = row(arena, new float[8]);
            hadamard.forward(v, rotated, 1, 8);
            hadamard.inverse(rotated, 1, 8);
            assertArrayEquals(x, read(rotated), 1e-6f);
        }
    }

    @Test
    void forwardIsTheNormalizedSylvesterTransformOfTheSignedInput() {
        float[] x = {1, 2, 3, 4, 5, 6, 7, 8};
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> out = row(arena, new float[8]);
            hadamard.forward(row(arena, x), out, 1, 8);
            // D·x = {1,-2,3,4 | -5,6,-7,-8}; H4 rows ++++, +-+-, ++--, +--+, halved
            assertArrayEquals(new float[] {3, 1, -4, 2, -7, -5, 8, -6}, read(out), 1e-6f);
        }
    }

    @Test
    void deltaOutRegroupsTiledHeadsBeforeTheTransform() {
        float[] tiled = {
            1, 2, 3, 4, 5, 6, 7, 8
        }; // [hd, nk, rep]: heads (k0,r0) (k1,r0) (k0,r1) (k1,r1)
        float[] grouped = {
            1, 2, 5, 6, 3, 4, 7, 8
        }; // [hd, rep, nk]: (k0,r0) (k0,r1) (k1,r0) (k1,r1)
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> a = row(arena, new float[8]), b = row(arena, new float[8]);
            hadamard.forwardDeltaOut(row(arena, tiled), a, 1, 8);
            hadamard.forward(row(arena, grouped), b, 1, 8);
            assertArrayEquals(read(b), read(a), 0f);
        }
    }

    @Test
    void aFullBlockMatchesTheDefinition() { // the vector path: 512-element blocks, 2 per row, 3
        // rows
        int block = 512, width = 1024, rows = 3;
        Random rng = new Random(7);
        int[] sign = new int[width];
        for (int i = 0; i < width; i++) sign[i] = rng.nextBoolean() ? 1 : -1;
        Hadamard h = new Hadamard(block, Map.of(width, sign), new int[] {width}, 2, 2, 1);
        float[] x = new float[rows * width];
        for (int i = 0; i < x.length; i++) x[i] = (float) rng.nextGaussian();
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> in =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), rows, width);
            MemoryView<MemorySegment> out =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), rows, width);
            MemorySegment.copy(
                    x, 0, in.memory().base(), ValueLayout.JAVA_FLOAT, in.byteOffset(), x.length);
            h.forward(in, out, rows, width);
            float[] y = new float[rows * width];
            MemorySegment.copy(
                    out.memory().base(), ValueLayout.JAVA_FLOAT, out.byteOffset(), y, 0, y.length);
            for (int r = 0; r < rows; r++)
                for (int b = 0; b < width; b += block)
                    for (int i = 0; i < block; i++) {
                        double expected = 0;
                        for (int j = 0; j < block; j++) {
                            int e = r * width + b + j;
                            expected +=
                                    (Integer.bitCount(i & j) % 2 == 0 ? 1 : -1)
                                            * sign[b + j]
                                            * x[e];
                        }
                        expected /= Math.sqrt(block);
                        assertEquals(
                                expected,
                                y[r * width + b + i],
                                1e-4,
                                "row " + r + " at " + (b + i));
                    }
        }
    }

    private static MemoryView<MemorySegment> row(Arena arena, float[] values) {
        MemoryView<MemorySegment> v =
                Views.allocateF32(MemoryAllocators.ofArena(arena), 1, values.length);
        MemorySegment.copy(
                values,
                0,
                v.memory().base(),
                ValueLayout.JAVA_FLOAT,
                v.byteOffset(),
                values.length);
        return v;
    }

    private static float[] read(MemoryView<MemorySegment> v) {
        float[] out = new float[Math.toIntExact(v.shape().flatAt(1))];
        MemorySegment.copy(
                v.memory().base(), ValueLayout.JAVA_FLOAT, v.byteOffset(), out, 0, out.length);
        return out;
    }
}
