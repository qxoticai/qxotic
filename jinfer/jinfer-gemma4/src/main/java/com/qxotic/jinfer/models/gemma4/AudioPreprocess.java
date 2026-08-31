package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Parallel;
import com.qxotic.jinfer.media.Media;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** PCM conversion shared by Gemma 4 audio projectors. */
public final class AudioPreprocess {
    public static final int SAMPLE_RATE = 16_000;
    public static final int N_FFT = 512;
    public static final int WINDOW = 320;
    public static final int HOP = 160;
    public static final int CHUNK_SAMPLES = 30 * SAMPLE_RATE;
    static final int N_BINS = N_FFT / 2 + 1;
    static final float MEL_FLOOR = 0.001f;

    public record MelChunk(float[] data, int frames) {}

    private final int nMel;
    private final float[] hann;
    private final float[] melFilterbank;
    private final int[] bandStart, bandEnd;

    public AudioPreprocess(int nMel) {
        if (nMel <= 0) throw new IllegalArgumentException("nMel must be positive");
        this.nMel = nMel;
        this.hann = buildHann();
        this.melFilterbank = buildMelFilterbank(nMel);
        this.bandStart = new int[nMel];
        this.bandEnd = new int[nMel];
        for (int m = 0; m < nMel; m++) {
            int first = N_BINS, last = -1;
            for (int b = 0; b < N_BINS; b++) {
                if (melFilterbank[m * N_BINS + b] != 0f) {
                    if (first == N_BINS) first = b;
                    last = b;
                }
            }
            bandStart[m] = first == N_BINS ? 0 : first & ~3;
            bandEnd[m] = last + 1;
        }
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
        int paddedNeeded = (frames - 1) * HOP + N_FFT;
        int totalPad = Math.max(paddedNeeded - length, WINDOW / 2);
        float[] padded = new float[totalPad + length];
        System.arraycopy(pcm, from, padded, WINDOW / 2, length);
        float[] output = new float[frames * nMel];
        int threads = Parallel.threads();
        float[][] fftInputs = new float[threads][N_FFT * 2];
        float[][] fftOutputs = new float[threads][N_FFT * 8];
        float[][] magnitudes = new float[threads][N_BINS];
        Parallel.forLoop(
                0,
                frames,
                (t, slot) -> {
                    float[] fftInput = fftInputs[slot];
                    float[] fftOutput = fftOutputs[slot];
                    float[] magnitude = magnitudes[slot];
                    int offset = t * HOP;
                    for (int k = 0; k < WINDOW; k++) fftInput[k] = hann[k] * padded[offset + k];
                    fftReal(fftInput, 0, N_FFT, fftOutput, 0);
                    for (int b = 0; b < N_BINS; b++) {
                        float power =
                                fftOutput[2 * b] * fftOutput[2 * b]
                                        + fftOutput[2 * b + 1] * fftOutput[2 * b + 1];
                        magnitude[b] = (float) Math.sqrt(power);
                    }
                    int lastGroup = (N_BINS - 1) & ~3;
                    for (int m = 0; m < nMel; m++) {
                        double sum = 0;
                        int base = m * N_BINS;
                        int end = Math.min(bandEnd[m], lastGroup);
                        for (int b = bandStart[m]; b < end; b += 4)
                            sum +=
                                    magnitude[b] * melFilterbank[base + b]
                                            + magnitude[b + 1] * melFilterbank[base + b + 1]
                                            + magnitude[b + 2] * melFilterbank[base + b + 2]
                                            + magnitude[b + 3] * melFilterbank[base + b + 3];
                        for (int b = Math.max(bandStart[m], lastGroup); b < bandEnd[m]; b++)
                            sum += magnitude[b] * melFilterbank[base + b];
                        output[t * nMel + m] = (float) Math.log(Math.max(sum, MEL_FLOOR));
                    }
                });
        return new MelChunk(output, frames);
    }

    private static float[] buildHann() {
        float[] window = new float[N_FFT];
        float pi = (float) Math.PI;
        for (int i = 0; i < WINDOW; i++) {
            float argument = (2f * pi * i) / WINDOW;
            window[i] = 0.5f - 0.5f * (float) Math.cos(argument);
        }
        return window;
    }

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

    private static final float[] SIN = new float[N_FFT];
    private static final float[] COS = new float[N_FFT];

    static {
        for (int i = 0; i < N_FFT; i++) {
            double theta = 2 * Math.PI * i / N_FFT;
            SIN[i] = (float) Math.sin((double) (float) theta);
            COS[i] = (float) Math.cos((double) (float) theta);
        }
    }

    static void fftReal(float[] input, int inputOffset, int n, float[] output, int outputOffset) {
        if (n == 1) {
            output[outputOffset] = input[inputOffset];
            output[outputOffset + 1] = 0;
            return;
        }
        int half = n / 2;
        int childInputOffset = inputOffset + n;
        for (int i = 0; i < half; i++) input[childInputOffset + i] = input[inputOffset + 2 * i];
        int evenOutputOffset = outputOffset + 2 * n;
        fftReal(input, childInputOffset, half, output, evenOutputOffset);
        for (int i = 0; i < half; i++) input[childInputOffset + i] = input[inputOffset + 2 * i + 1];
        int oddOutputOffset = evenOutputOffset + n;
        fftReal(input, childInputOffset, half, output, oddOutputOffset);
        int step = N_FFT / n;
        for (int k = 0; k < half; k++) {
            int index = k * step;
            float real = COS[index], imaginary = -SIN[index];
            float oddReal = output[oddOutputOffset + 2 * k];
            float oddImaginary = output[oddOutputOffset + 2 * k + 1];
            output[outputOffset + 2 * k] =
                    output[evenOutputOffset + 2 * k] + real * oddReal - imaginary * oddImaginary;
            output[outputOffset + 2 * k + 1] =
                    output[evenOutputOffset + 2 * k + 1]
                            + real * oddImaginary
                            + imaginary * oddReal;
            output[outputOffset + 2 * (k + half)] =
                    output[evenOutputOffset + 2 * k] - real * oddReal + imaginary * oddImaginary;
            output[outputOffset + 2 * (k + half) + 1] =
                    output[evenOutputOffset + 2 * k + 1]
                            - real * oddImaginary
                            - imaginary * oddReal;
        }
    }
}
