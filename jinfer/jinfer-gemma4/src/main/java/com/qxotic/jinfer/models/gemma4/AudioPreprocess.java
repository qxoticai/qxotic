package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.kernels.LogMel;
import com.qxotic.jinfer.media.Media;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** PCM conversion shared by Gemma 4 audio projectors. */
final class AudioPreprocess {
    public static final int SAMPLE_RATE = 16_000;
    public static final int N_FFT = 512;
    public static final int WINDOW = 320;
    public static final int HOP = 160;
    public static final int CHUNK_SAMPLES = 30 * SAMPLE_RATE;
    static final int N_BINS = N_FFT / 2 + 1;

    /** Added before the log, {@code log(mel + 0.001)}, as the reference does (llama.cpp clamps). */
    static final float MEL_FLOOR = 0.001f;

    public record MelChunk(float[] data, int frames) {}

    private final int nMel;
    private final LogMel logMel;

    public AudioPreprocess(int nMel) {
        if (nMel <= 0) throw new IllegalArgumentException("nMel must be positive");
        this.nMel = nMel;
        this.logMel =
                new LogMel(
                        new LogMel.Spec(
                                N_FFT,
                                HOP,
                                nMel,
                                buildHann(),
                                buildMelFilterbank(nMel),
                                0f,
                                1f,
                                0f,
                                MEL_FLOOR));
    }

    /**
     * Number of samples produced by {@link #toMono16k(Media.Audio)}: miniaudio's {@code
     * ma_calculate_frame_count_after_resampling}, the buffer llama.cpp decodes into.
     */
    public static int mono16kLength(Media.Audio audio) {
        Objects.requireNonNull(audio, "audio");
        int channels = audio.channels();
        int sampleRate = audio.sampleRate();
        if (channels <= 0) throw new IllegalArgumentException("audio channels must be positive");
        if (sampleRate <= 0)
            throw new IllegalArgumentException("audio sampleRate must be positive");
        long frames = audio.pcm().length / channels;
        if (sampleRate == SAMPLE_RATE || frames == 0) return Math.toIntExact(frames);
        long out = frames * SAMPLE_RATE / sampleRate;
        long fromFraction = out * (sampleRate / SAMPLE_RATE) / SAMPLE_RATE;
        if (out * (sampleRate % SAMPLE_RATE) + fromFraction <= frames) out++;
        return Math.toIntExact(out);
    }

    /**
     * Averages interleaved channels, then resamples to 16 kHz as llama.cpp's audio decoder does
     * (miniaudio's linear resampler): see {@link #resample}.
     */
    public static float[] toMono16k(Media.Audio audio) {
        int outputLength = mono16kLength(audio);
        int channels = audio.channels();
        float[] input = audio.pcm();
        int frames = input.length / channels;
        float[] mono;
        if (channels == 1) {
            mono = input;
        } else {
            mono = new float[frames];
            for (int frame = 0; frame < frames; frame++) {
                float sum = 0;
                for (int channel = 0; channel < channels; channel++)
                    sum += input[frame * channels + channel];
                mono[frame] = sum / channels;
            }
        }
        if (audio.sampleRate() == SAMPLE_RATE) return mono;
        return resample(mono, audio.sampleRate(), outputLength);
    }

    /**
     * miniaudio's linear resampler, op for op: the rates reduced by their gcd, a 4th-order
     * Butterworth low-pass (two float biquads, transposed direct form II, cut at the lower Nyquist)
     * on the input side when downsampling and the output side when upsampling, and linear
     * interpolation on a rational clock that loads one input before the first output (a one-input
     * delay). Output stops when the input runs out; the rest of the {@code length} stays zero.
     */
    static float[] resample(float[] input, int sampleRate, int length) {
        int gcd = gcd(sampleRate, SAMPLE_RATE);
        int rateIn = sampleRate / gcd, rateOut = SAMPLE_RATE / gcd;
        boolean down = rateIn > rateOut;
        float[][] lowPass = new float[2][];
        float[][] state = new float[2][2];
        double w = 2 * Math.PI * (Math.min(rateIn, rateOut) * 0.5) / Math.max(rateIn, rateOut);
        double c = Math.cos(w), s = Math.sin(w);
        for (int i = 0; i < 2; i++) {
            double q = 1 / (2 * Math.cos((1 + i * 2) * (Math.PI / (4 * 2))));
            double alpha = s / (2 * q), a0 = 1 + alpha;
            lowPass[i] =
                    new float[] {
                        (float) ((1 - c) / 2 / a0),
                        (float) ((1 - c) / a0),
                        (float) ((1 - c) / 2 / a0),
                        (float) (-2 * c / a0),
                        (float) ((1 - alpha) / a0)
                    };
        }
        float[] output = new float[length];
        float x0 = 0, x1 = 0;
        long timeInt = 1, timeFraction = 0;
        for (int o = 0, read = 0; o < length; o++) {
            for (; timeInt > 0 && read < input.length; timeInt--) {
                x0 = x1;
                x1 = down ? filter(lowPass, state, input[read++]) : input[read++];
            }
            if (timeInt > 0) break;
            float a = (float) timeFraction / rateOut;
            float y = x0 + (x1 - x0) * a;
            output[o] = down ? y : filter(lowPass, state, y);
            timeInt += rateIn / rateOut;
            timeFraction += rateIn % rateOut;
            if (timeFraction >= rateOut) {
                timeFraction -= rateOut;
                timeInt++;
            }
        }
        return output;
    }

    /** The biquad cascade over one sample: {@code {b0, b1, b2, a1, a2}} per stage. */
    private static float filter(float[][] stages, float[][] state, float x) {
        for (int i = 0; i < stages.length; i++) {
            float[] k = stages[i];
            float y = k[0] * x + state[i][0];
            state[i][0] = k[1] * x - k[3] * y + state[i][1];
            state[i][1] = k[2] * x - k[4] * y;
            x = y;
        }
        return x;
    }

    private static int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    public static int framesFor(int chunkSamples) {
        return (chunkSamples + WINDOW / 2 - (WINDOW + 1)) / HOP + 1;
    }

    public List<MelChunk> logMel(Media.Audio audio) {
        return logMel(toMono16k(audio));
    }

    public List<MelChunk> logMel(float[] pcm) {
        List<MelChunk> chunks = new ArrayList<>();
        for (int offset = 0; offset < pcm.length; offset += CHUNK_SAMPLES)
            chunks.add(chunk(pcm, offset, Math.min(CHUNK_SAMPLES, pcm.length - offset)));
        return chunks;
    }

    private MelChunk chunk(float[] pcm, int from, int length) {
        int frames = framesFor(length);
        if (frames == 0) return new MelChunk(new float[0], 0);
        return new MelChunk(logMel.frames(pcm, from, length, WINDOW / 2, frames), frames);
    }

    /** Periodic Hann of {@link #WINDOW} samples, left-aligned and zero-padded to the FFT size. */
    private static float[] buildHann() {
        float[] window = new float[N_FFT];
        float pi = (float) Math.PI;
        for (int i = 0; i < WINDOW; i++) {
            float argument = (2f * pi * i) / WINDOW;
            window[i] = 0.5f - 0.5f * (float) Math.cos(argument);
        }
        return window;
    }

    /** Triangular HTK-scale filterbank, unnormalized, as the reference builds it. */
    private static float[] buildMelFilterbank(int nMel) {
        double low = hzToMel(0), high = hzToMel(0.5 * SAMPLE_RATE);
        double[] hz = new double[nMel + 2];
        for (int i = 0; i < hz.length; i++)
            hz[i] = melToHz(low + (high - low) * ((double) i / (nMel + 1)));
        double binHz = (double) SAMPLE_RATE / N_FFT;
        float[] filterbank = new float[nMel * N_BINS];
        for (int m = 0; m < nMel; m++) {
            double left = hz[m], center = hz[m + 1], right = hz[m + 2];
            double leftWidth = Math.max(1e-30, center - left);
            double rightWidth = Math.max(1e-30, right - center);
            for (int b = 0; b < N_BINS; b++) {
                double frequency = b * binHz;
                double weight = 0;
                if (frequency >= left && frequency <= center)
                    weight = (frequency - left) / leftWidth;
                else if (frequency > center && frequency <= right)
                    weight = (right - frequency) / rightWidth;
                filterbank[m * N_BINS + b] = (float) weight;
            }
        }
        return filterbank;
    }

    private static double hzToMel(double hz) {
        return 2595 * Math.log10(1 + hz / 700);
    }

    private static double melToHz(double mel) {
        return 700 * (Math.pow(10, mel / 2595) - 1);
    }
}
