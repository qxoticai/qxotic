package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

class Gemma4ConformerOpsTest {

    private final Arena arena = Arena.ofAuto();

    @Test
    void gluUsesFirstHalfAsValueAndSecondHalfAsGate() {
        float[] packed = {99f, -2f, 0f, 3f, 4f, -5f, -1f, 0f, 1f, 20f, 88f};
        MemorySegment out = f32(new float[6]);

        Activations.glu(view(out, 6), 1, view(f32(packed), packed.length), 1, 4);

        for (int i = 0; i < 4; i++) {
            float expected = packed[1 + i] * FastMath.sigmoid(packed[5 + i]);
            assertEquals(expected, get(out, 1 + i), 0f, "lane " + i);
        }
        assertEquals(0f, get(out, 0), 0f);
        assertEquals(0f, get(out, 5), 0f);
    }

    @Test
    void dotMatchesScalarSpanOracle() {
        float[] a = new float[23], b = new float[23];
        for (int i = 0; i < a.length; i++) {
            a[i] = i - 9f;
            b[i] = 7f - i;
        }
        float expected = 0f;
        for (int i = 2; i < 21; i++) expected += a[i] * b[i];

        assertEquals(
                expected, Ops.dot(view(f32(a), a.length), 2, view(f32(b), b.length), 2, 19), 1e-6f);
    }

    @Test
    void stride2Pad1Conv2dMatchesScalarOracleForOddShape() {
        int time = 3, frequency = 5, inChannels = 2, outChannels = 2;
        float[] input = new float[inChannels * time * frequency];
        float[] taps = new float[outChannels * inChannels * 9];
        for (int i = 0; i < input.length; i++) input[i] = i * 0.25f - 2f;
        for (int i = 0; i < taps.length; i++) taps[i] = (i % 7 - 3) * 0.125f;
        int timeOut = 2, frequencyOut = 3;
        MemorySegment actual = f32(new float[outChannels * timeOut * frequencyOut]);

        Convolutions.conv2dStride2Pad1(
                view(f32(input), input.length),
                time,
                frequency,
                inChannels,
                taps,
                outChannels,
                view(actual, outChannels * timeOut * frequencyOut));

        for (int oc = 0; oc < outChannels; oc++) {
            for (int ot = 0; ot < timeOut; ot++) {
                for (int of = 0; of < frequencyOut; of++) {
                    float expected = 0f;
                    for (int ic = 0; ic < inChannels; ic++) {
                        for (int ky = 0; ky < 3; ky++) {
                            int it = 2 * ot - 1 + ky;
                            if (it < 0 || it >= time) continue;
                            for (int kx = 0; kx < 3; kx++) {
                                int inf = 2 * of - 1 + kx;
                                if (inf < 0 || inf >= frequency) continue;
                                expected +=
                                        taps[((oc * inChannels + ic) * 3 + ky) * 3 + kx]
                                                * input[(ic * time + it) * frequency + inf];
                            }
                        }
                    }
                    int at = (oc * timeOut + ot) * frequencyOut + of;
                    assertEquals(expected, get(actual, at), 0f, "output " + at);
                }
            }
        }
    }

    @Test
    void channelLayerNormReluMatchesScalarOracle() {
        int channels = 3, positions = 4;
        float[] values = {-3f, 1f, 4f, 2f, 0f, -2f, 5f, 2f, 6f, 3f, -1f, 2f};
        float[] original = values.clone();
        float[] weights = {0.5f, -1.5f, 2f};
        MemorySegment actual = f32(values);

        Norms.layerNormChannelsReluInPlace(
                view(actual, values.length),
                view(f32(weights), weights.length),
                channels,
                positions,
                1e-6f);

        for (int p = 0; p < positions; p++) {
            double mean = 0;
            for (int c = 0; c < channels; c++) mean += original[c * positions + p];
            mean /= channels;
            double variance = 0;
            for (int c = 0; c < channels; c++) {
                double d = original[c * positions + p] - mean;
                variance += d * d;
            }
            float inv = (float) (1.0 / Math.sqrt(variance / channels + 1e-6f));
            for (int c = 0; c < channels; c++) {
                float expected =
                        Math.max(
                                0f,
                                (float) ((original[c * positions + p] - mean) * inv) * weights[c]);
                assertEquals(
                        expected,
                        get(actual, c * positions + p),
                        0f,
                        "channel/position " + c + "/" + p);
            }
        }
    }

    @Test
    void causalDepthwiseTemporalConvolutionMatchesScalarOracle() {
        int time = 5, channels = 3, kernel = 3;
        float[] input = new float[time * channels];
        float[] taps = {1f, 2f, 3f, -1f, 0.5f, 2f, 0.25f, -0.5f, 1.5f};
        for (int i = 0; i < input.length; i++) input[i] = i - 4f;
        MemorySegment actual = f32(new float[input.length]);

        Convolutions.causalDepthwise1d(
                view(f32(input), input.length),
                view(f32(taps), taps.length),
                view(actual, input.length),
                time,
                channels,
                kernel);

        for (int t = 0; t < time; t++) {
            for (int c = 0; c < channels; c++) {
                float expected = 0f;
                for (int k = 0; k < kernel; k++) {
                    int source = t - kernel + 1 + k;
                    if (source >= 0)
                        expected += taps[c * kernel + k] * input[source * channels + c];
                }
                assertEquals(
                        expected, get(actual, t * channels + c), 0f, "time/channel " + t + "/" + c);
            }
        }
    }

    private MemorySegment f32(float[] values) {
        MemorySegment segment = arena.allocate((long) values.length * Float.BYTES, 64);
        for (int i = 0; i < values.length; i++) {
            segment.set(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) i * Float.BYTES, values[i]);
        }
        return segment;
    }

    private static float get(MemorySegment segment, int index) {
        return segment.get(ValueLayout.JAVA_FLOAT_UNALIGNED, (long) index * Float.BYTES);
    }

    private static MemoryView<MemorySegment> view(MemorySegment segment, int size) {
        return Oracles.f32View(segment, size);
    }
}
