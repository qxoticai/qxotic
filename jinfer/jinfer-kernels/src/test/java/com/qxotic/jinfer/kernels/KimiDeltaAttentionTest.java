package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

class KimiDeltaAttentionTest {
    @Test
    void preparesLegacyUnsafeGatesWithSoftplus() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var gate = Views.allocateF32(memory, 4);
            var beta = Views.allocateF32(memory, 2);
            KimiDeltaAttention.gates(
                    Views.fromFloatArray(memory, new float[] {-2f, 0f, 1f, 3f}),
                    Views.fromFloatArray(memory, new float[] {-1f, 2f}),
                    Views.fromFloatArray(memory, new float[] {.5f, -.5f, .5f, -.5f}),
                    Views.fromFloatArray(memory, new float[] {-2f, -3f}),
                    gate,
                    beta,
                    1,
                    2,
                    2,
                    false,
                    -5f);
            assertArrayEquals(
                    new float[] {
                        -2f * softplus(-1.5f),
                        -2f * softplus(-.5f),
                        -3f * softplus(1.5f),
                        -3f * softplus(2.5f)
                    },
                    Views.toFloatArray(gate, "gate"),
                    1e-6f);
            assertArrayEquals(
                    new float[] {Activations.sigmoid(-1f), Activations.sigmoid(2f)},
                    Views.toFloatArray(beta, "beta"),
                    1e-6f);
        }
    }

    @Test
    void vectorDecayScanMatchesTheScalarRecurrence() {
        assertScanMatchesOracle(3, 2, 3, 1e-6f);
    }

    @Test
    void productionShapeVectorScanMatchesTheScalarRecurrence() {
        assertScanMatchesOracle(2, 2, 128, 2e-5f);
    }

    private static void assertScanMatchesOracle(int rows, int heads, int dim, float tolerance) {
        float[] q = values(rows * heads * dim, .13f, .02f);
        float[] k = values(rows * heads * dim, .17f, -.03f);
        float[] v = values(rows * heads * dim, .19f, .04f);
        float[] gate = values(rows * heads * dim, .07f, -1.2f);
        float[] beta = values(rows * heads, .11f, .45f);
        float[] initial = values(heads * dim * dim, .05f, .01f);
        float[] expectedState = initial.clone();
        float[] expected = oracle(q, k, v, gate, beta, expectedState, rows, heads, dim);
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var state = Views.fromFloatArray(memory, initial);
            var out = Views.allocateF32(memory, q.length);
            KimiDeltaAttention.scan(
                    Views.fromFloatArray(memory, q),
                    Views.fromFloatArray(memory, k),
                    Views.fromFloatArray(memory, v),
                    Views.fromFloatArray(memory, gate),
                    Views.fromFloatArray(memory, beta),
                    state,
                    out,
                    Views.allocateF32(memory, heads * dim),
                    rows,
                    heads,
                    dim);
            assertArrayEquals(expected, Views.toFloatArray(out, "out"), tolerance);
            assertArrayEquals(expectedState, Views.toFloatArray(state, "state"), tolerance);
        }
    }

    private static float[] oracle(
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
        for (int head = 0; head < heads; head++) {
            int sb = head * dim * dim;
            for (int row = 0; row < rows; row++) {
                int base = (row * heads + head) * dim;
                for (int j = 0; j < dim; j++) {
                    float sk = 0f;
                    for (int i = 0; i < dim; i++) {
                        int at = sb + j * dim + i;
                        state[at] *= (float) Math.exp(gate[base + i]);
                        sk += state[at] * k[base + i];
                    }
                    float delta = (v[base + j] - sk) * beta[row * heads + head];
                    for (int i = 0; i < dim; i++) {
                        int at = sb + j * dim + i;
                        state[at] += delta * k[base + i];
                        out[base + j] += state[at] * q[base + i];
                    }
                    out[base + j] *= 1.0f / (float) Math.sqrt(dim);
                }
            }
        }
        return out;
    }

    private static float[] values(int size, float frequency, float offset) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) values[i] = .1f * (float) Math.sin(i * frequency) + offset;
        return values;
    }

    private static float softplus(float x) {
        return (float) Math.log1p(Math.exp(x));
    }
}
