package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class Mamba2Test {
    @Test
    void scanAndGroupedNormMatchScalarOracle() {
        int rows = 2, inner = 4, heads = 2, groups = 1, stateSize = 2;
        int channels = inner + 2 * groups * stateSize;
        try (Arena arena = Arena.ofConfined()) {
            PanamaMemoryArena memory = new PanamaMemoryArena(arena);
            MemoryView<MemorySegment> conv = Views.allocateF32(memory, rows, channels);
            MemoryView<MemorySegment> z = Views.allocateF32(memory, rows, inner);
            MemoryView<MemorySegment> dt = Views.allocateF32(memory, rows, heads);
            MemoryView<MemorySegment> a = Views.allocateF32(memory, heads);
            MemoryView<MemorySegment> d = Views.allocateF32(memory, heads);
            MemoryView<MemorySegment> state = Views.allocateF32(memory, inner, stateSize);
            MemoryView<MemorySegment> output = Views.allocateF32(memory, rows, inner);
            MemoryView<MemorySegment> norm = Views.allocateF32(memory, inner);
            MemoryView<MemorySegment> normalized = Views.allocateF32(memory, rows, inner);
            float[] cv = {
                .2f, -.1f, .3f, .4f, .5f, -.2f, .1f, .3f, -.3f, .2f, .1f, -.4f, .25f, .15f, -.2f,
                .35f
            };
            float[] zv = {.1f, .2f, -.3f, .4f, -.1f, .3f, .2f, -.2f};
            float[] tv = {.2f, .3f, .1f, .25f};
            float[] av = {-.5f, -.25f}, dv = {.2f, -.1f}, sv = new float[inner * stateSize];
            Views.copyFromArray(conv, 0, cv, 0, cv.length, "conv");
            Views.copyFromArray(z, 0, zv, 0, zv.length, "z");
            Views.copyFromArray(dt, 0, tv, 0, tv.length, "dt");
            Views.copyFromArray(a, 0, av, 0, av.length, "a");
            Views.copyFromArray(d, 0, dv, 0, dv.length, "d");
            Views.copyFromArray(norm, 0, new float[] {1, 2, 3, 4}, 0, inner, "norm");
            float[] expected = new float[rows * inner];
            int headDim = inner / heads, qSize = groups * stateSize;
            for (int row = 0; row < rows; row++)
                for (int head = 0; head < heads; head++) {
                    float decay = (float) Math.exp(tv[row * heads + head] * av[head]);
                    for (int lane = 0; lane < headDim; lane++) {
                        int index = head * headDim + lane, st = index * stateSize;
                        float x = cv[row * channels + index], sum = 0f;
                        for (int i = 0; i < stateSize; i++) {
                            sv[st + i] =
                                    sv[st + i] * decay
                                            + cv[row * channels + inner + i]
                                                    * x
                                                    * tv[row * heads + head];
                            sum += sv[st + i] * cv[row * channels + inner + qSize + i];
                        }
                        expected[row * inner + index] =
                                (sum + x * dv[head]) * Activations.silu(zv[row * inner + index]);
                    }
                }
            Mamba2.scan(conv, z, dt, a, d, state, output, rows, inner, heads, groups, stateSize);
            for (int i = 0; i < expected.length; i++)
                assertEquals(expected[i], Views.getFloat(output, i, "output"));
            Mamba2.groupedRmsNorm(output, norm, normalized, rows, inner, groups, 1e-5f);
            for (int row = 0; row < rows; row++) {
                float sum = 0f;
                for (int i = 0; i < inner; i++)
                    sum += expected[row * inner + i] * expected[row * inner + i];
                float inv = (float) (1.0 / Math.sqrt(sum / inner + 1e-5f));
                for (int i = 0; i < inner; i++)
                    assertEquals(
                            expected[row * inner + i] * inv * (i + 1),
                            Views.getFloat(normalized, row * inner + i, "normalized"));
            }
        }
    }
}
