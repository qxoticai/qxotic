package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

/** Direct contracts for the stateful causal convolution used by recurrent models. */
class ConvolutionsTest {

    private final Arena arena = Arena.ofAuto();

    @Test
    void statefulCausalDepthwiseSiluMatchesScalarOracle() {
        float[] input = {1, 2, 3, 4, 5, 6};
        float[] taps = {0.5f, -1, 2, -0.25f, 0.75f, 1.5f};
        float[] initial = {-2, -1, 0.25f, 0.5f};
        float[] expected = new float[input.length];
        int rows = 3, channels = 2, kernel = 3, hist = 2;
        for (int row = 0; row < rows; row++) {
            for (int c = 0; c < channels; c++) {
                float sum = 0;
                for (int k = 0; k < kernel; k++) {
                    int pos = row - hist + k;
                    sum +=
                            taps[c * kernel + k]
                                    * (pos < 0
                                            ? initial[(pos + hist) * channels + c]
                                            : input[pos * channels + c]);
                }
                expected[row * channels + c] = (float) (sum / (1 + Math.exp(-sum)));
            }
        }
        MemorySegment in = arena.allocate(4L * input.length, 64);
        MemorySegment w = arena.allocate(4L * taps.length, 64);
        in.copyFrom(MemorySegment.ofArray(input));
        w.copyFrom(MemorySegment.ofArray(taps));
        MemorySegment state = arena.allocate(4L * initial.length, 64), out = arena.allocate(24, 64);
        state.copyFrom(MemorySegment.ofArray(initial));
        Convolutions.causalDepthwiseSilu(
                Oracles.f32View(in, input.length),
                Oracles.f32View(w, taps.length),
                Oracles.f32View(state, initial.length),
                Oracles.f32View(out, input.length),
                rows,
                channels,
                kernel);
        assertArrayEquals(expected, out.toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT), 1e-6f);
        assertArrayEquals(
                new float[] {3, 4, 5, 6}, state.toArray(java.lang.foreign.ValueLayout.JAVA_FLOAT));
    }

    @Test
    void biasedStatefulCausalDepthwiseSiluMatchesScalarOracle() {
        int rows = 3, channels = 2, kernel = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> input =
                    Views.allocateF32(
                            new com.qxotic.jinfer.PanamaMemoryArena(arena), rows, channels);
            MemoryView<MemorySegment> taps =
                    Views.allocateF32(
                            new com.qxotic.jinfer.PanamaMemoryArena(arena), channels, kernel);
            MemoryView<MemorySegment> bias =
                    Views.allocateF32(new com.qxotic.jinfer.PanamaMemoryArena(arena), channels);
            MemoryView<MemorySegment> history =
                    Views.allocateF32(
                            new com.qxotic.jinfer.PanamaMemoryArena(arena), kernel - 1, channels);
            MemoryView<MemorySegment> output =
                    Views.allocateF32(
                            new com.qxotic.jinfer.PanamaMemoryArena(arena), rows, channels);
            Views.copyFromArray(input, 0, new float[] {1, 2, 3, 4, 5, 6}, 0, 6, "input");
            Views.copyFromArray(taps, 0, new float[] {.1f, .2f, .3f, -.2f, .4f, .1f}, 0, 6, "taps");
            Views.copyFromArray(bias, 0, new float[] {.5f, -.25f}, 0, 2, "bias");
            Views.copyFromArray(history, 0, new float[] {-1, -2, -.5f, -1.5f}, 0, 4, "history");
            float[] oldHistory = Views.toFloatArray(history, "history");
            float[] values = Views.toFloatArray(input, "input");
            float[] weights = Views.toFloatArray(taps, "taps");
            float[] biases = Views.toFloatArray(bias, "bias");
            Convolutions.causalDepthwiseSilu(
                    input, taps, bias, history, output, rows, channels, kernel);
            for (int row = 0; row < rows; row++)
                for (int channel = 0; channel < channels; channel++) {
                    float sum = biases[channel];
                    for (int k = 0; k < kernel; k++) {
                        int pos = row - kernel + 1 + k;
                        float value =
                                pos < 0
                                        ? oldHistory[(pos + kernel - 1) * channels + channel]
                                        : values[pos * channels + channel];
                        sum += weights[channel * kernel + k] * value;
                    }
                    assertEquals(
                            Activations.silu(sum),
                            Views.getFloat(output, row * channels + channel, "output"));
                }
        }
    }
}
