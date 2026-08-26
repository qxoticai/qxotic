package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.F_SPECIES;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class Mamba2Test {
    @Test
    void vectorScanAndGroupedNormMatchScalarAtNemotronShape() {
        assumeTrue(F_SPECIES != null && 128 % F_SPECIES.length() == 0);
        int rows = 17, inner = 128, heads = 4, groups = 2, stateSize = 128;
        int channels = inner + 2 * groups * stateSize;
        try (Arena arena = Arena.ofConfined()) {
            float[] convValues = values(rows * channels, .013f, .02f);
            float[] zValues = values(rows * inner, .017f, -.03f);
            float[] dtValues = values(rows * heads, .019f, .15f);
            float[] aValues = values(heads, .071f, -.45f);
            float[] dValues = values(heads, .053f, .1f);
            float[] stateValues = values(inner * stateSize, .007f, .01f);
            float[] weightValues = values(inner, .031f, 1f);

            var conv = view(arena, convValues);
            var z = view(arena, zValues);
            var dt = view(arena, dtValues);
            var a = view(arena, aValues);
            var d = view(arena, dValues);
            var scalarState = view(arena, stateValues);
            var vectorState = view(arena, stateValues);
            var chunkedState = view(arena, stateValues);
            var scalarOut = view(arena, new float[rows * inner]);
            var vectorOut = view(arena, new float[rows * inner]);
            var chunkedOut = view(arena, new float[rows * inner]);

            Mamba2.scanScalar(
                    Raw.f32(conv, "conv"),
                    Raw.f32(z, "z"),
                    Raw.f32(dt, "dt"),
                    Raw.f32(a, "a"),
                    Raw.f32(d, "d"),
                    Raw.f32(scalarState, "state"),
                    Raw.f32(scalarOut, "output"),
                    rows,
                    inner,
                    heads,
                    groups,
                    stateSize);
            VectorMamba2.scan(
                    Raw.f32(conv, "conv"),
                    Raw.f32(z, "z"),
                    Raw.f32(dt, "dt"),
                    Raw.f32(a, "a"),
                    Raw.f32(d, "d"),
                    Raw.f32(vectorState, "state"),
                    Raw.f32(vectorOut, "output"),
                    rows,
                    inner,
                    heads,
                    groups,
                    stateSize);

            int split = 5;
            vectorScan(
                    rows(conv, 0, split, channels),
                    rows(z, 0, split, inner),
                    rows(dt, 0, split, heads),
                    a,
                    d,
                    chunkedState,
                    rows(chunkedOut, 0, split, inner),
                    split,
                    inner,
                    heads,
                    groups,
                    stateSize);
            vectorScan(
                    rows(conv, split, rows, channels),
                    rows(z, split, rows, inner),
                    rows(dt, split, rows, heads),
                    a,
                    d,
                    chunkedState,
                    rows(chunkedOut, split, rows, inner),
                    rows - split,
                    inner,
                    heads,
                    groups,
                    stateSize);

            assertArrayEquals(floats(scalarOut), floats(vectorOut), 3e-5f);
            assertArrayEquals(floats(scalarState), floats(vectorState), 3e-5f);
            assertArrayEquals(floats(vectorOut), floats(chunkedOut), 0f);
            assertArrayEquals(floats(vectorState), floats(chunkedState), 0f);

            var weight = view(arena, weightValues);
            var scalarNorm = view(arena, new float[rows * inner]);
            var vectorNorm = view(arena, new float[rows * inner]);
            var chunkedNorm = view(arena, new float[rows * inner]);
            Mamba2.groupedRmsNormScalar(
                    Raw.f32(scalarOut, "input"),
                    Raw.f32(weight, "weight"),
                    Raw.f32(scalarNorm, "output"),
                    rows,
                    inner,
                    groups,
                    1e-5f);
            VectorMamba2.groupedRmsNorm(
                    Raw.f32(scalarOut, "input"),
                    Raw.f32(weight, "weight"),
                    Raw.f32(vectorNorm, "output"),
                    rows,
                    inner,
                    groups,
                    1e-5f);
            VectorMamba2.groupedRmsNorm(
                    Raw.f32(rows(scalarOut, 0, split, inner), "input"),
                    Raw.f32(weight, "weight"),
                    Raw.f32(rows(chunkedNorm, 0, split, inner), "output"),
                    split,
                    inner,
                    groups,
                    1e-5f);
            VectorMamba2.groupedRmsNorm(
                    Raw.f32(rows(scalarOut, split, rows, inner), "input"),
                    Raw.f32(weight, "weight"),
                    Raw.f32(rows(chunkedNorm, split, rows, inner), "output"),
                    rows - split,
                    inner,
                    groups,
                    1e-5f);
            assertArrayEquals(floats(scalarNorm), floats(vectorNorm), 3e-5f);
            assertArrayEquals(floats(vectorNorm), floats(chunkedNorm), 0f);
        }
    }

    @Test
    void scanAndGroupedNormMatchScalarOracle() {
        int rows = 2, inner = 4, heads = 2, groups = 1, stateSize = 2;
        int channels = inner + 2 * groups * stateSize;
        try (Arena arena = Arena.ofConfined()) {
            MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(arena);
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

    private static void vectorScan(
            MemoryView<MemorySegment> conv,
            MemoryView<MemorySegment> z,
            MemoryView<MemorySegment> dt,
            MemoryView<MemorySegment> a,
            MemoryView<MemorySegment> d,
            MemoryView<MemorySegment> state,
            MemoryView<MemorySegment> output,
            int rows,
            int inner,
            int heads,
            int groups,
            int stateSize) {
        VectorMamba2.scan(
                Raw.f32(conv, "conv"),
                Raw.f32(z, "z"),
                Raw.f32(dt, "dt"),
                Raw.f32(a, "a"),
                Raw.f32(d, "d"),
                Raw.f32(state, "state"),
                Raw.f32(output, "output"),
                rows,
                inner,
                heads,
                groups,
                stateSize);
    }

    private static MemoryView<MemorySegment> rows(
            MemoryView<MemorySegment> view, int start, int end, int rowSize) {
        return view.slice(0, (long) start * rowSize, (long) end * rowSize);
    }

    private static MemoryView<MemorySegment> view(Arena arena, float[] values) {
        MemorySegment segment = arena.allocate(4L * values.length, 64);
        segment.copyFrom(MemorySegment.ofArray(values));
        return Oracles.f32View(segment, values.length);
    }

    private static float[] floats(MemoryView<MemorySegment> view) {
        return view.memory()
                .base()
                .asSlice(view.byteOffset(), view.logicalSize() * Float.BYTES)
                .toArray(ValueLayout.JAVA_FLOAT);
    }

    private static float[] values(int size, float frequency, float offset) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) result[i] = .1f * (float) Math.sin(i * frequency) + offset;
        return result;
    }
}
