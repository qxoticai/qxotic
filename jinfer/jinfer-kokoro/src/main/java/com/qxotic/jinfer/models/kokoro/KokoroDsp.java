package com.qxotic.jinfer.models.kokoro;

import com.qxotic.jota.memory.MemoryAllocator;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Random;

/** Host-side source generation and spectral transforms for Kokoro's iSTFTNet. */
final class KokoroDsp {

    private static final int UPSAMPLE = 300;
    private static final int HARMONICS = 9;
    private static final int SAMPLE_RATE = 24_000;
    private static final int FFT_SIZE = 20;
    private static final int HOP_SIZE = 5;
    private static final int BINS = FFT_SIZE / 2 + 1;
    private static final float TWO_PI = (float) (2 * Math.PI);
    private static final float SINE_AMPLITUDE = 0.1f;
    private static final float NOISE_STDDEV = 0.003f;
    private static final float VOICED_THRESHOLD = 10;
    private static final float[] HANN = hann();

    private KokoroDsp() {}

    /** Magnitude and phase are both channel-major {@code [11][frames]}. */
    record Spectrum(float[][] magnitude, float[][] phase) {}

    /**
     * Generates Kokoro's nine-harmonic HnNSF source and returns its centered STFT. The F0 input has
     * length {@code 2*T}; the source linear layer has nine weights.
     */
    static Spectrum sourceStft(
            float[] f0,
            float[] linearWeight,
            float linearBias,
            Random random,
            MemoryAllocator<MemorySegment> scratch) {
        require(f0.length > 0 && (f0.length & 1) == 0, "F0 must have positive length 2*T");
        require(linearWeight.length == HARMONICS, "source linear weight must have 9 values");
        require(random != null, "random must not be null");

        int highLength = Math.multiplyExact(f0.length, UPSAMPLE);
        for (float value : f0) require(Float.isFinite(value), "F0 must be finite");
        // The reference adds these offsets before centered 300:1 interpolation, which never
        // samples index zero. Keep consuming them so seeded Gaussian noise remains unchanged.
        for (int harmonic = 1; harmonic < HARMONICS; harmonic++) random.nextFloat();

        float[][] phaseLow = KokoroWorkspace.takeMatrix(scratch, HARMONICS, f0.length);
        for (int harmonic = 0; harmonic < HARMONICS; harmonic++) {
            float sum = 0;
            for (int t = 0; t < f0.length; t++) {
                float cycles = f0[t] * (harmonic + 1) / SAMPLE_RATE;
                sum += cycles - (float) Math.floor(cycles);
                phaseLow[harmonic][t] = sum * TWO_PI;
            }
        }

        float[] source = KokoroWorkspace.takeFloats(scratch, highLength);
        for (int t = 0; t < highLength; t++) {
            float voiced = f0[t / UPSAMPLE] > VOICED_THRESHOLD ? 1 : 0;
            float noiseAmplitude = voiced != 0 ? NOISE_STDDEV : SINE_AMPLITUDE / 3;
            float mixed = linearBias;
            float sourceIndex = (t + 0.5f) / UPSAMPLE - 0.5f;
            for (int harmonic = 0; harmonic < HARMONICS; harmonic++) {
                float phase = interpolate(phaseLow[harmonic], sourceIndex) * UPSAMPLE;
                float sine = (float) Math.sin(phase) * SINE_AMPLITUDE;
                float wave = sine * voiced + noiseAmplitude * (float) random.nextGaussian();
                mixed += wave * linearWeight[harmonic];
            }
            source[t] = (float) Math.tanh(mixed);
        }
        return stft(source, scratch);
    }

    /**
     * Inverts generator magnitude and phase in channel-major {@code [11][frames]} layout. The
     * returned centered waveform has exactly {@code (frames - 1) * 5} samples.
     */
    static float[] istft(
            float[][] magnitude, float[][] phase, MemoryAllocator<MemorySegment> scratch) {
        int frames = spectralFrames(magnitude, "magnitude");
        require(spectralFrames(phase, "phase") == frames, "magnitude and phase frames differ");
        int outputLength = Math.multiplyExact(frames - 1, HOP_SIZE);
        if (outputLength == 0) return new float[0];

        int paddedLength = outputLength + FFT_SIZE;
        float[] waveform = KokoroWorkspace.takeFloats(scratch, paddedLength);
        float[] windowSum = KokoroWorkspace.takeFloats(scratch, paddedLength);
        Arrays.fill(waveform, 0);
        Arrays.fill(windowSum, 0);
        for (int frame = 0; frame < frames; frame++) {
            int offset = frame * HOP_SIZE;
            for (int n = 0; n < FFT_SIZE; n++) {
                // DC is the magnitude itself, irrespective of its supplied phase.
                float value = magnitude[0][frame];
                float nyquist =
                        magnitude[BINS - 1][frame] * (float) Math.cos(phase[BINS - 1][frame]);
                value += (n & 1) == 0 ? nyquist : -nyquist;
                for (int bin = 1; bin < BINS - 1; bin++) {
                    float angle = TWO_PI * bin * n / FFT_SIZE;
                    float real = magnitude[bin][frame] * (float) Math.cos(phase[bin][frame]);
                    float imaginary = magnitude[bin][frame] * (float) Math.sin(phase[bin][frame]);
                    value +=
                            2
                                    * (real * (float) Math.cos(angle)
                                            - imaginary * (float) Math.sin(angle));
                }
                float window = HANN[n];
                waveform[offset + n] += value / FFT_SIZE * window;
                windowSum[offset + n] += window * window;
            }
        }

        float[] output = new float[outputLength];
        for (int i = 0; i < outputLength; i++) {
            float divisor = windowSum[FFT_SIZE / 2 + i];
            output[i] = divisor > 1e-11f ? waveform[FFT_SIZE / 2 + i] / divisor : 0;
        }
        return output;
    }

    private static Spectrum stft(float[] source, MemoryAllocator<MemorySegment> scratch) {
        int pad = FFT_SIZE / 2;
        float[] padded = KokoroWorkspace.takeFloats(scratch, source.length + 2 * pad);
        System.arraycopy(source, 0, padded, pad, source.length);
        for (int i = 0; i < pad; i++) {
            padded[pad - 1 - i] = source[i + 1];
            padded[pad + source.length + i] = source[source.length - 2 - i];
        }

        int frames = source.length / HOP_SIZE + 1;
        float[][] magnitude = KokoroWorkspace.takeMatrix(scratch, BINS, frames);
        float[][] phase = KokoroWorkspace.takeMatrix(scratch, BINS, frames);
        for (int frame = 0; frame < frames; frame++) {
            for (int bin = 0; bin < BINS; bin++) {
                float real = 0;
                float imaginary = 0;
                for (int n = 0; n < FFT_SIZE; n++) {
                    float sample = padded[frame * HOP_SIZE + n] * HANN[n];
                    float angle = -TWO_PI * bin * n / FFT_SIZE;
                    real += sample * (float) Math.cos(angle);
                    imaginary += sample * (float) Math.sin(angle);
                }
                magnitude[bin][frame] = (float) Math.sqrt(real * real + imaginary * imaginary);
                phase[bin][frame] = (float) Math.atan2(imaginary, real);
            }
        }
        return new Spectrum(magnitude, phase);
    }

    private static float interpolate(float[] values, float index) {
        float clamped = Math.max(0, Math.min(values.length - 1, index));
        int lower = (int) Math.floor(clamped);
        int upper = Math.min(lower + 1, values.length - 1);
        float fraction = clamped - lower;
        return values[lower] * (1 - fraction) + values[upper] * fraction;
    }

    private static int spectralFrames(float[][] values, String name) {
        require(values.length == BINS, name + " must have 11 bins");
        int frames = values[0].length;
        require(frames > 0, name + " must have at least one frame");
        for (float[] bin : values) require(bin.length == frames, name + " must be rectangular");
        return frames;
    }

    private static float[] hann() {
        float[] window = new float[FFT_SIZE];
        for (int i = 0; i < FFT_SIZE; i++)
            window[i] = 0.5f - 0.5f * (float) Math.cos(TWO_PI * i / FFT_SIZE);
        return window;
    }

    private static void require(boolean condition, String message) {
        if (!condition) throw new IllegalArgumentException("Kokoro: " + message);
    }
}
