package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.PanamaMemoryArena;
import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** Formula and boundary contracts independent of the retired FloatTensor implementation. */
class NormsContractTest {

    @Test
    void rmsNormMatchesItsDefinitionAcrossTheVectorTail() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = new PanamaMemoryArena(arena);
            float[] values = new float[35];
            float[] weights = new float[33];
            for (int i = 0; i < values.length; i++) values[i] = (i % 9) - 4.25f;
            for (int i = 0; i < weights.length; i++) weights[i] = 0.5f + i / 64f;
            var x = Views.fromFloatArray(memory, values);
            var weight = Views.fromFloatArray(memory, weights);
            var out = Views.allocateF32(memory, values.length);

            Norms.rmsnorm(out, 1, x, 2, weight, 33, 1e-5f);

            double squares = 0;
            for (int i = 0; i < 33; i++) squares += values[i + 2] * values[i + 2];
            float scale = (float) (1 / Math.sqrt(squares / 33 + 1e-5));
            float[] actual = Views.toFloatArray(out, "out");
            for (int i = 0; i < 33; i++) {
                assertEquals(values[i + 2] * weights[i] * scale, actual[i + 1], 2e-6f, "lane " + i);
            }
        }
    }

    @Test
    void layerNormMatchesItsDefinitionPerRow() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = new PanamaMemoryArena(arena);
            float[] values = {1, 2, 5, 8, -2, 0, 4, 6};
            float[] gamma = {1, 0.5f, 2, -1};
            float[] beta = {0, 1, -2, 3};
            var x = Views.fromFloatArray(memory, values);
            var out = Views.allocateF32(memory, values.length);

            Norms.layerNorm(
                    out,
                    x,
                    Views.fromFloatArray(memory, gamma),
                    Views.fromFloatArray(memory, beta),
                    4,
                    2,
                    1e-5f);

            float[] expected = new float[values.length];
            for (int row = 0; row < 2; row++) {
                float mean = 0;
                for (int c = 0; c < 4; c++) mean += values[row * 4 + c];
                mean /= 4;
                float variance = 0;
                for (int c = 0; c < 4; c++) {
                    float d = values[row * 4 + c] - mean;
                    variance += d * d;
                }
                float inv = (float) (1 / Math.sqrt(variance / 4 + 1e-5));
                for (int c = 0; c < 4; c++) {
                    expected[row * 4 + c] = (values[row * 4 + c] - mean) * inv * gamma[c] + beta[c];
                }
            }
            assertArrayEquals(expected, Views.toFloatArray(out, "out"), 1e-6f);
        }
    }

    @Test
    void rejectsWrongDatatype() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = new PanamaMemoryArena(arena);
            var f32 = Views.allocateF32(memory, 4);
            var f16 = Views.wrap(arena.allocate(8), DataType.FP16, Shape.flat(4));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Norms.rmsnorm(f32, 0, f16, 0, f32, 4, 1e-5f));
        }
    }
}
