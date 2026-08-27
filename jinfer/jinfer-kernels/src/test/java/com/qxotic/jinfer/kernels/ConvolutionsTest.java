package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

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
        assertArrayEquals(expected, out.toArray(ValueLayout.JAVA_FLOAT), 1e-6f);
        assertArrayEquals(new float[] {3, 4, 5, 6}, state.toArray(ValueLayout.JAVA_FLOAT));
    }

    /**
     * {@link Convolutions#conv1dRows} against a scalar reference, over shapes that reach every
     * path: full 4-channel groups and a partial group, the branch-free interior at each register
     * tile ({@code -Djinfer.convTile}, see {@link KernelSelectionTest}), both edges, dilation, an
     * even kernel (asymmetric "same" padding) and a time axis shorter than one tile span.
     */
    @ParameterizedTest(name = "in={0} out={1} time={2} kernel={3} dilation={4}")
    @CsvSource({
        "3, 8, 300, 5, 1",
        "3, 8, 300, 5, 2",
        "2, 9, 257, 11, 1",
        "2, 4, 300, 4, 1",
        "2, 4, 300, 4, 3",
        "1, 4, 7, 3, 1",
        "5, 1, 130, 1, 1",
    })
    void conv1dRowsMatchesScalarOracle(
            int inChannels, int outChannels, int time, int kernel, int dilation) {
        Random rnd = new Random(31L * inChannels + 7L * outChannels + time + kernel + dilation);
        float[] input = new float[inChannels * time];
        float[] taps = new float[outChannels * inChannels * kernel];
        float[] bias = new float[outChannels];
        for (int i = 0; i < input.length; i++) input[i] = rnd.nextFloat() * 2 - 1;
        for (int i = 0; i < taps.length; i++) taps[i] = rnd.nextFloat() * 2 - 1;
        for (int i = 0; i < bias.length; i++) bias[i] = rnd.nextFloat() * 2 - 1;

        int pad = ((kernel - 1) * dilation) / 2;
        float[] expected = new float[outChannels * time];
        for (int oc = 0; oc < outChannels; oc++) {
            for (int t = 0; t < time; t++) {
                double sum = bias[oc];
                for (int ic = 0; ic < inChannels; ic++) {
                    for (int k = 0; k < kernel; k++) {
                        int pos = t + k * dilation - pad;
                        if (pos < 0 || pos >= time) continue;
                        sum += taps[(oc * inChannels + ic) * kernel + k] * input[ic * time + pos];
                    }
                }
                expected[oc * time + t] = (float) sum;
            }
        }

        try (Arena arena = Arena.ofConfined()) {
            var memory = MemoryAllocators.ofArena(arena);
            MemoryView<MemorySegment> in = Views.allocateF32(memory, inChannels, time);
            MemoryView<MemorySegment> out = Views.allocateF32(memory, outChannels, time);
            MemoryView<MemorySegment> biasView = Views.allocateF32(memory, outChannels);
            Views.copyFromArray(in, 0, input, 0, input.length, "in");
            Views.copyFromArray(biasView, 0, bias, 0, bias.length, "bias");
            Convolutions.conv1dRows(
                    in, inChannels, out, outChannels, time, kernel, dilation, taps, biasView);
            float[] actual = Views.toFloatArray(out, "out");
            for (int i = 0; i < expected.length; i++) {
                assertEquals(
                        expected[i],
                        actual[i],
                        1e-4f,
                        "channel "
                                + i / time
                                + " sample "
                                + i % time
                                + " (tile "
                                + Convolutions.tileCode()
                                + ")");
            }
        }
    }

    @Test
    void biasedStatefulCausalDepthwiseSiluMatchesScalarOracle() {
        int rows = 3, channels = 2, kernel = 3;
        try (Arena arena = Arena.ofConfined()) {
            MemoryView<MemorySegment> input =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), rows, channels);
            MemoryView<MemorySegment> taps =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), channels, kernel);
            MemoryView<MemorySegment> bias =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), channels);
            MemoryView<MemorySegment> history =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), kernel - 1, channels);
            MemoryView<MemorySegment> output =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), rows, channels);
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
