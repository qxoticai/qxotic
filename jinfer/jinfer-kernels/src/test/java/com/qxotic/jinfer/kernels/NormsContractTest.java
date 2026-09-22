package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryAllocators;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Test;

/** Formula and boundary contracts independent of the retired FloatTensor implementation. */
class NormsContractTest {

    @Test
    void rmsNormMatchesItsDefinitionAcrossTheVectorTail() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
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
    void ggmlRmsNormUsesFloatProductsAndDoubleAccumulation() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] values = new float[2048];
            float[] weights = new float[values.length];
            values[0] = 4096f;
            for (int i = 1; i < values.length; i++) values[i] = 1f;
            for (int i = 0; i < weights.length; i++) weights[i] = 0.75f + i / 8192f;
            var out = Views.allocateF32(memory, values.length);

            Norms.rmsnormGgml(
                    out,
                    0,
                    Views.fromFloatArray(memory, values),
                    0,
                    Views.fromFloatArray(memory, weights),
                    values.length,
                    1e-6f);

            double sum = 0;
            for (float value : values) sum += (double) (value * value);
            float mean = (float) (sum / values.length);
            float scale = 1f / (float) Math.sqrt(mean + 1e-6f);
            float[] actual = Views.toFloatArray(out, "out");
            for (int i = 0; i < values.length; i++)
                assertEquals((values[i] * scale) * weights[i], actual[i], "lane " + i);
        }
    }

    @Test
    void layerNormMatchesItsDefinitionPerRow() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            float[] values = {1, 2, 5, 8, -2, 0, 4, 6};
            float[] gamma = {1, 0.5f, 2, -1};
            float[] beta = {0, 1, -2, 3};
            var x = Views.fromFloatArray(memory, values);
            var out = Views.allocateF32(memory, values.length);

            Norms.layerNormRows(
                    out,
                    x,
                    Views.fromFloatArray(memory, gamma),
                    Views.fromFloatArray(memory, beta),
                    2,
                    4,
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

    /**
     * Rows split across the pool, a width off every vector multiple (scalar tails), and values near
     * 1000 with unit spread: a one-pass {@code E[x²] - mean²} variance is off by ~3% there. The
     * two-pass one is within 5e-3 on both paths; the scalar path's serial float mean (sums near
     * 1e6) is the looser, at ~1.3e-3.
     */
    @Test
    void layerNormRowsIsStableForLargeMeans() {
        int rows = 37, width = 1027;
        java.util.Random random = new java.util.Random(7);
        float[] values = new float[rows * width];
        float[] gamma = new float[width], beta = new float[width];
        for (int i = 0; i < values.length; i++) values[i] = 1000 + (float) random.nextGaussian();
        for (int c = 0; c < width; c++) {
            gamma[c] = 0.5f + random.nextFloat();
            beta[c] = random.nextFloat() - 0.5f;
        }
        try (Arena arena = Arena.ofShared()) {
            var memory = MemoryAllocators.ofArena(arena);
            var x = Views.fromFloatArray(memory, values);
            Norms.layerNormRows(
                    x,
                    x, // in place
                    Views.fromFloatArray(memory, gamma),
                    Views.fromFloatArray(memory, beta),
                    rows,
                    width,
                    1e-5f);
            float[] actual = Views.toFloatArray(x, "out");
            for (int row = 0; row < rows; row++) {
                double mean = 0, variance = 0;
                for (int c = 0; c < width; c++) mean += values[row * width + c];
                mean /= width;
                for (int c = 0; c < width; c++)
                    variance += Math.pow(values[row * width + c] - mean, 2);
                double inv = 1 / Math.sqrt(variance / width + 1e-5);
                for (int c = 0; c < width; c++) {
                    double expected = (values[row * width + c] - mean) * inv * gamma[c] + beta[c];
                    assertEquals(
                            expected, actual[row * width + c], 5e-3, "row " + row + " lane " + c);
                }
            }
        }
    }

    @Test
    void rejectsWrongDatatype() {
        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            var f32 = Views.allocateF32(memory, 4);
            var f16 = Views.wrap(arena.allocate(8), DataType.FP16, Shape.flat(4));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> Norms.rmsnorm(f32, 0, f16, 0, f32, 4, 1e-5f));
        }
    }
}
