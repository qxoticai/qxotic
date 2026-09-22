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
                                MEL_FLOOR,
                                0f));
    }

    /** Number of samples produced by {@link #toMono16k(Media.Audio)}. */
    public static int mono16kLength(Media.Audio audio) {
        Objects.requireNonNull(audio, "audio");
        int channels = audio.channels();
        int sampleRate = audio.sampleRate();
        if (channels <= 0) throw new IllegalArgumentException("audio channels must be positive");
        if (sampleRate <= 0)
            throw new IllegalArgumentException("audio sampleRate must be positive");
        int frames = audio.pcm().length / channels;
        if (sampleRate == SAMPLE_RATE) return frames;
        return Math.max(
                1, Math.toIntExact(Math.round(frames * ((double) SAMPLE_RATE / sampleRate))));
    }

    /** Averages interleaved channels, then linearly resamples to 16 kHz. */
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

        float[] output = new float[outputLength];
        if (mono.length == 0) return output;
        double ratio = (double) SAMPLE_RATE / audio.sampleRate();
        for (int i = 0; i < output.length; i++) {
            double sourcePosition = i / ratio;
            int source = (int) sourcePosition;
            double fraction = sourcePosition - source;
            float a = mono[Math.min(source, mono.length - 1)];
            float b = mono[Math.min(source + 1, mono.length - 1)];
            output[i] = (float) (a + (b - a) * fraction);
        }
        return output;
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

    /** Triangular HTK-scale filterbank, unnormalized, matching llama.cpp's gemma4a projector. */
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
