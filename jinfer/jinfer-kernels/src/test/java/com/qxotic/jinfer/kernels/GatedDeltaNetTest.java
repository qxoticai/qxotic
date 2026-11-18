package com.qxotic.jinfer.kernels;

import static com.qxotic.jinfer.Segments.F_SPECIES;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.Segments;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class GatedDeltaNetTest {
    /**
     * The scalar-geometry cliff blames the geometry only: on a slow Vector API JIT (stock C2, or
     * {@code -Djinfer.vector.jit=slow}) every geometry runs scalar and that is SLOW_JIT's report.
     */
    @Test
    void geometryCliffOnlyWhereTheVectorPathIsAvailable() {
        boolean available = Segments.FAST_VECTOR_JIT && VectorGatedDeltaNet.enabled();
        assertFalse(GatedDeltaNet.geometryUncovered(256), "a covered geometry");
        assertEquals(
                available,
                GatedDeltaNet.geometryUncovered(F_SPECIES == null ? 3 : F_SPECIES.length() + 1));
        if (!Segments.FAST_VECTOR_JIT)
            assertFalse(GatedDeltaNet.geometryUncovered(3), "a slow JIT");
    }

    @Test
    void vectorScanMatchesScalarAtQwen35Shape() {
        assumeTrue(F_SPECIES != null && 128 % F_SPECIES.length() == 0);
        try (Arena arena = Arena.ofConfined()) {
            int rows = 17, heads = 2, dim = 128;
            float[] q = values(rows * heads * dim, .013f, .17f);
            float[] k = values(rows * heads * dim, .017f, -.11f);
            float[] v = values(rows * heads * dim, .019f, .07f);
            float[] gate = values(rows * heads, .023f, -.2f);
            float[] beta = values(rows * heads, .029f, .5f);
            float[] initialState = values(heads * dim * dim, .003f, .01f);

            var qv = view(arena, q);
            var kv = view(arena, k);
            var vv = view(arena, v);
            var gv = view(arena, gate);
            var bv = view(arena, beta);
            var scalarState = view(arena, initialState);
            var vectorState = view(arena, initialState);
            var scalarOut = view(arena, new float[q.length]);
            var vectorOut = view(arena, new float[q.length]);
            var scalarSk = view(arena, new float[heads * dim]);
            var vectorSk = view(arena, new float[heads * dim]);
            var scalarDelta = view(arena, new float[heads * dim]);
            var vectorDelta = view(arena, new float[heads * dim]);

            GatedDeltaNet.scanScalar(
                    Raw.f32(qv, "q"),
                    Raw.f32(kv, "k"),
                    Raw.f32(vv, "v"),
                    Raw.f32(gv, "gate"),
                    Raw.f32(bv, "beta"),
                    Raw.f32(scalarState, "state"),
                    Raw.f32(scalarOut, "output"),
                    Raw.f32(scalarSk, "sk"),
                    Raw.f32(scalarDelta, "delta"),
                    rows,
                    heads,
                    dim);
            VectorGatedDeltaNet.scan(
                    Raw.f32(qv, "q"),
                    Raw.f32(kv, "k"),
                    Raw.f32(vv, "v"),
                    Raw.f32(gv, "gate"),
                    Raw.f32(bv, "beta"),
                    Raw.f32(vectorState, "state"),
                    Raw.f32(vectorOut, "output"),
                    Raw.f32(vectorSk, "sk"),
                    Raw.f32(vectorDelta, "delta"),
                    rows,
                    heads,
                    dim);

            assertArrayEquals(floats(scalarOut), floats(vectorOut), 2e-5f);
            assertArrayEquals(floats(scalarState), floats(vectorState), 2e-5f);
            assertArrayEquals(floats(scalarSk), floats(vectorSk), 2e-5f);
            assertArrayEquals(floats(scalarDelta), floats(vectorDelta), 2e-5f);
        }
    }

    /**
     * Strong decays: log-gates down to -8 make a chunk's cumulative decay underflow, which is why
     * chunked formulations of this scan were rejected (see the design note in git history,
     * bench/DeltaNetParity). The recurrence must stay finite and the vector path must still match
     * the scalar oracle there.
     */
    @Test
    void strongDecaysStayFiniteAndInAgreement() {
        assumeTrue(F_SPECIES != null && 128 % F_SPECIES.length() == 0);
        try (Arena arena = Arena.ofConfined()) {
            int rows = 64, heads = 2, dim = 128;
            float[] q = values(rows * heads * dim, .013f, .17f);
            float[] k = values(rows * heads * dim, .017f, -.11f);
            float[] v = values(rows * heads * dim, .019f, .07f);
            float[] gate = new float[rows * heads];
            for (int i = 0; i < gate.length; i++) gate[i] = -8f * (i % 7) / 6f; // 0 .. -8
            float[] beta = values(rows * heads, .029f, .5f);
            float[] initialState = values(heads * dim * dim, .003f, .01f);

            var qv = view(arena, q);
            var kv = view(arena, k);
            var vv = view(arena, v);
            var gv = view(arena, gate);
            var bv = view(arena, beta);
            var scalarState = view(arena, initialState);
            var vectorState = view(arena, initialState);
            var scalarOut = view(arena, new float[q.length]);
            var vectorOut = view(arena, new float[q.length]);
            var sk = view(arena, new float[heads * dim]);
            var delta = view(arena, new float[heads * dim]);

            GatedDeltaNet.scanScalar(
                    Raw.f32(qv, "q"),
                    Raw.f32(kv, "k"),
                    Raw.f32(vv, "v"),
                    Raw.f32(gv, "gate"),
                    Raw.f32(bv, "beta"),
                    Raw.f32(scalarState, "state"),
                    Raw.f32(scalarOut, "output"),
                    Raw.f32(sk, "sk"),
                    Raw.f32(delta, "delta"),
                    rows,
                    heads,
                    dim);
            VectorGatedDeltaNet.scan(
                    Raw.f32(qv, "q"),
                    Raw.f32(kv, "k"),
                    Raw.f32(vv, "v"),
                    Raw.f32(gv, "gate"),
                    Raw.f32(bv, "beta"),
                    Raw.f32(vectorState, "state"),
                    Raw.f32(vectorOut, "output"),
                    Raw.f32(sk, "sk"),
                    Raw.f32(delta, "delta"),
                    rows,
                    heads,
                    dim);

            float[] out = floats(scalarOut);
            float[] state = floats(scalarState);
            for (float value : out) assertTrue(Float.isFinite(value));
            for (float value : state) assertTrue(Float.isFinite(value));
            assertArrayEquals(out, floats(vectorOut), 2e-5f);
            assertArrayEquals(state, floats(vectorState), 2e-5f);
        }
    }

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

    /** Per-head RMSNorm times SiLU(z) against a double-precision oracle, vector body and tail. */
    @Test
    void postNormMatchesOracle() {
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3, heads = 2, dim = 37;
            float[] x = values(rows * heads * dim, .7f, .05f);
            float[] z = values(rows * heads * dim, 1.3f, -.02f);
            float[] w = values(dim, .4f, 1f);
            MemorySegment out = arena.allocate(4L * x.length, 64);
            GatedDeltaNet.postNorm(
                    view(arena, x),
                    view(arena, z),
                    view(arena, w),
                    Oracles.f32View(out, x.length),
                    rows,
                    heads,
                    dim,
                    1e-6f);
            float[] actual = out.toArray(ValueLayout.JAVA_FLOAT);
            for (int base = 0; base < x.length; base += dim) {
                double ss = 0;
                for (int d = 0; d < dim; d++) ss += (double) x[base + d] * x[base + d];
                double inv = 1 / Math.sqrt(ss / dim + 1e-6);
                for (int d = 0; d < dim; d++) {
                    double zv = z[base + d];
                    double expected = x[base + d] * inv * w[d] * zv / (1 + Math.exp(-zv));
                    assertEquals(expected, actual[base + d], 1e-5 * Math.abs(expected) + 1e-7);
                }
            }
        }
    }

    /**
     * {@code values *= sigmoid(gate)} against a double-precision oracle, vector body and tail,
     * gates out to +-12: the error stays within 2e-5 of the gated value (the tanh form's absolute
     * error), however small sigmoid itself gets.
     */
    @Test
    void sigmoidMultiplyMatchesOracle() {
        try (Arena arena = Arena.ofConfined()) {
            int rows = 3, dim = 37;
            float[] x = values(rows * dim, .7f, .05f);
            float[] g = new float[rows * dim];
            for (int i = 0; i < g.length; i++) g[i] = 12f * (float) Math.sin(i * .37f);
            MemorySegment out = arena.allocate(4L * x.length, 64);
            out.copyFrom(MemorySegment.ofArray(x));
            GatedDeltaNet.sigmoidMultiply(
                    Oracles.f32View(out, x.length), view(arena, g), rows, dim);
            float[] actual = out.toArray(ValueLayout.JAVA_FLOAT);
            for (int i = 0; i < x.length; i++) {
                double expected = x[i] / (1 + Math.exp(-g[i]));
                assertEquals(expected, actual[i], 2e-5 * Math.abs(x[i]) + 1e-9);
            }
        }
    }

    private static float[] values(int size, float frequency, float offset) {
        float[] result = new float[size];
        for (int i = 0; i < size; i++) result[i] = .1f * (float) Math.sin(i * frequency) + offset;
        return result;
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
