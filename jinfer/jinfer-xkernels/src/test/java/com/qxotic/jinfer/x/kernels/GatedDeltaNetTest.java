package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class GatedDeltaNetTest {
    @Test
    void scanMatchesScalarOracleAcrossChunkBoundary() {
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3, heads = 2, dim = 2;
            float[] q = {.2f, -.4f, .6f, .1f, -.3f, .7f, .5f, -.2f, .8f, .3f, -.1f, .9f};
            float[] k = {.4f, .2f, -.5f, .3f, .1f, .8f, -.4f, .6f, .7f, -.2f, .2f, .5f};
            float[] v = {.3f, -.1f, .9f, .2f, -.6f, .4f, .1f, .7f, .5f, -.8f, .6f, .3f};
            float[] gate = {-.1f, -.2f, -.3f, -.4f, -.5f, -.6f};
            float[] beta = {.2f, .3f, .4f, .5f, .6f, .7f};
            float[] expectedState = new float[heads * dim * dim];
            float[] expected = scalar(q, k, v, gate, beta, expectedState, rows, heads, dim);
            var qv = view(arena, q);
            var kv = view(arena, k);
            var vv = view(arena, v);
            var gv = view(arena, gate);
            var bv = view(arena, beta);
            MemorySegment state = arena.allocate(4L * expectedState.length, 64);
            MemorySegment out = arena.allocate(4L * expected.length, 64);
            MemorySegment sk = arena.allocate(4L * heads * dim, 64),
                    delta = arena.allocate(4L * heads * dim, 64);
            GatedDeltaNet.scan(
                    qv,
                    kv,
                    vv,
                    gv,
                    bv,
                    Oracles.f32View(state, expectedState.length),
                    Oracles.f32View(out, expected.length),
                    Oracles.f32View(sk, heads * dim),
                    Oracles.f32View(delta, heads * dim),
                    1,
                    heads,
                    dim);
            GatedDeltaNet.scan(
                    qv.slice(0, heads * dim, rows * heads * dim),
                    kv.slice(0, heads * dim, rows * heads * dim),
                    vv.slice(0, heads * dim, rows * heads * dim),
                    gv.slice(0, heads, rows * heads),
                    bv.slice(0, heads, rows * heads),
                    Oracles.f32View(state, expectedState.length),
                    Oracles.f32View(out, expected.length).slice(0, heads * dim, rows * heads * dim),
                    Oracles.f32View(sk, heads * dim),
                    Oracles.f32View(delta, heads * dim),
                    rows - 1,
                    heads,
                    dim);
            assertArrayEquals(expected, out.toArray(ValueLayout.JAVA_FLOAT), 1e-6f);
            assertArrayEquals(expectedState, state.toArray(ValueLayout.JAVA_FLOAT), 1e-6f);
        }
    }

    private static com.qxotic.jota.memory.MemoryView<MemorySegment> view(
            Arena arena, float[] values) {
        MemorySegment segment = arena.allocate(4L * values.length, 64);
        segment.copyFrom(MemorySegment.ofArray(values));
        return Oracles.f32View(segment, values.length);
    }

    private static float[] scalar(
            float[] q,
            float[] k,
            float[] v,
            float[] gate,
            float[] beta,
            float[] state,
            int rows,
            int heads,
            int dim) {
        float[] out = new float[q.length];
        for (int h = 0; h < heads; h++) {
            int sb = h * dim * dim;
            for (int r = 0; r < rows; r++) {
                int base = (r * heads + h) * dim;
                float decay = (float) Math.exp(gate[r * heads + h]);
                for (int i = 0; i < dim * dim; i++) state[sb + i] *= decay;
                float[] sk = new float[dim];
                for (int j = 0; j < dim; j++)
                    for (int d = 0; d < dim; d++) sk[j] += state[sb + j * dim + d] * k[base + d];
                for (int j = 0; j < dim; j++) {
                    float delta = (v[base + j] - sk[j]) * beta[r * heads + h];
                    for (int d = 0; d < dim; d++) state[sb + j * dim + d] += delta * k[base + d];
                }
                for (int j = 0; j < dim; j++)
                    for (int d = 0; d < dim; d++)
                        out[base + j] += state[sb + j * dim + d] * q[base + d];
            }
        }
        return out;
    }
}
