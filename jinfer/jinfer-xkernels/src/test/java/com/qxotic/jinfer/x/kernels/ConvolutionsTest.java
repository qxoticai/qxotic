package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracle: ported x.Convolutions.conv1dRows vs jinfer-core on identical inputs — full
 * groups and partial groups, interior/body/edge spans, dilations, with and without bias.
 */
class ConvolutionsTest {

    private final Arena arena = Arena.ofAuto();

    private void parity(
            int inCh, int outCh, int time, int kernel, int dilation, boolean withBias, long seed) {
        Random rng = new Random(seed);
        int inN = inCh * time, outN = outCh * time;
        MemorySegment in = Oracles.f32(arena, inN, seed);
        MemorySegment outOld = arena.allocate(4L * outN, 64);
        MemorySegment outNew = arena.allocate(4L * outN, 64);
        float[] taps = new float[outCh * inCh * kernel];
        for (int i = 0; i < taps.length; i++) taps[i] = rng.nextFloat() * 2 - 1;
        MemorySegment bias = withBias ? Oracles.f32(arena, outCh, seed + 1) : null;

        FloatTensor biasOld = withBias ? Oracles.oldF32(bias, outCh) : null;
        MemoryView<MemorySegment> biasNew = withBias ? Oracles.f32View(bias, outCh) : null;
        com.qxotic.jinfer.Convolutions.conv1dRows(
                (F32FloatTensor) Oracles.oldF32(in, inN),
                inCh,
                (F32FloatTensor) Oracles.oldF32(outOld, outN),
                outCh,
                time,
                kernel,
                dilation,
                taps,
                biasOld);
        Convolutions.conv1dRows(
                Oracles.f32View(in, inN),
                inCh,
                Oracles.f32View(outNew, outN),
                outCh,
                time,
                kernel,
                dilation,
                taps,
                biasNew);
        Oracles.assertClose(
                outOld,
                outNew,
                outN,
                String.format(
                        "conv %d->%d t=%d k=%d d=%d bias=%s",
                        inCh, outCh, time, kernel, dilation, withBias),
                1e-4);
    }

    @Test
    void fullGroupsWithBias() {
        parity(8, 8, 500, 11, 1, true, 1);
    }

    @Test
    void partialGroupNoBias() {
        parity(3, 5, 200, 3, 2, false, 2);
    }

    @Test
    void singleChannel() {
        parity(1, 1, 100, 5, 1, true, 3);
    }

    @Test
    void dilatedWideKernel() {
        parity(12, 4, 96, 11, 4, true, 4);
    }

    @Test
    void timeSmallerThanPad() {
        parity(2, 6, 8, 11, 3, false, 5);
    }

    @Test
    void largerTileSpan() {
        parity(4, 8, 9000, 3, 1, true, 6);
    }

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
                            new com.qxotic.jinfer.x.PanamaMemoryArena(arena), rows, channels);
            MemoryView<MemorySegment> taps =
                    Views.allocateF32(
                            new com.qxotic.jinfer.x.PanamaMemoryArena(arena), channels, kernel);
            MemoryView<MemorySegment> bias =
                    Views.allocateF32(new com.qxotic.jinfer.x.PanamaMemoryArena(arena), channels);
            MemoryView<MemorySegment> history =
                    Views.allocateF32(
                            new com.qxotic.jinfer.x.PanamaMemoryArena(arena), kernel - 1, channels);
            MemoryView<MemorySegment> output =
                    Views.allocateF32(
                            new com.qxotic.jinfer.x.PanamaMemoryArena(arena), rows, channels);
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
