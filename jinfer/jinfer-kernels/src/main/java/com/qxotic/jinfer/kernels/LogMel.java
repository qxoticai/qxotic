package com.qxotic.jinfer.kernels;

import com.qxotic.jinfer.Parallel;

/**
 * Log-mel spectrogram front end shared by audio models (Gemma 4 audio towers, Parakeet ASR).
 *
 * <p>The spec captures where model families disagree: window placement inside the FFT frame, signal
 * padding, optional pre-emphasis, magnitude versus power spectrum, and whether near-zero mel
 * energies are clamped ({@code melFloor}, Gemma) or offset ({@code logZeroGuard}, NeMo). The
 * filterbank may be computed by the caller or lifted verbatim from model weights.
 */
public final class LogMel {

    /**
     * @param nFft FFT size, a power of two; {@code window} must have this length, with the model's
     *     analysis window already placed at its position inside the frame (left-aligned for Gemma,
     *     centered for NeMo) and zeros elsewhere
     * @param hop frame step in samples
     * @param nMels mel bands; {@code filterbank} is {@code [nMels, nFft/2+1]} row-major
     * @param preemphasis {@code x[t] -= preemphasis * x[t-1]} on the raw signal ({@code <= 0} off)
     * @param magPower 1 applies the filterbank to spectral magnitudes, 2 to the power spectrum
     * @param melFloor lower clamp on the filterbank output before the log ({@code <= 0} off)
     * @param logZeroGuard value added to the filterbank output before the log ({@code <= 0} off)
     */
    public record Spec(
            int nFft,
            int hop,
            int nMels,
            float[] window,
            float[] filterbank,
            float preemphasis,
            float magPower,
            float melFloor,
            float logZeroGuard) {
        public Spec {
            if (nFft <= 0 || Integer.bitCount(nFft) != 1)
                throw new IllegalArgumentException("nFft must be a power of two, got " + nFft);
            if (hop <= 0) throw new IllegalArgumentException("hop must be positive");
            if (nMels <= 0) throw new IllegalArgumentException("nMels must be positive");
            if (window.length != nFft)
                throw new IllegalArgumentException(
                        "window length " + window.length + " != nFft " + nFft);
            if (filterbank.length != (long) nMels * (nFft / 2 + 1))
                throw new IllegalArgumentException(
                        "filterbank length "
                                + filterbank.length
                                + " != nMels * (nFft/2+1) = "
                                + nMels * (nFft / 2 + 1));
            if (magPower != 1f && magPower != 2f)
                throw new IllegalArgumentException("magPower must be 1 or 2, got " + magPower);
        }

        public int bins() {
            return nFft / 2 + 1;
        }
    }

    private final Spec spec;
    private final int bins;
    private final float[] sin, cos;
    private final int[] bandStart, bandEnd;

    public LogMel(Spec spec) {
        this.spec = spec;
        this.bins = spec.bins();
        this.sin = new float[spec.nFft()];
        this.cos = new float[spec.nFft()];
        for (int i = 0; i < spec.nFft(); i++) {
            double theta = 2 * Math.PI * i / spec.nFft();
            sin[i] = (float) Math.sin((double) (float) theta);
            cos[i] = (float) Math.cos((double) (float) theta);
        }
        // Filterbank rows are band-limited; skipping their exact-zero tails is bit-identical.
        this.bandStart = new int[spec.nMels()];
        this.bandEnd = new int[spec.nMels()];
        float[] filterbank = spec.filterbank();
        for (int m = 0; m < spec.nMels(); m++) {
            int first = bins, last = -1;
            for (int b = 0; b < bins; b++) {
                if (filterbank[m * bins + b] != 0f) {
                    if (first == bins) first = b;
                    last = b;
                }
            }
            bandStart[m] = first == bins ? 0 : first & ~3;
            bandEnd[m] = last + 1;
        }
    }

    /**
     * Frame-major log-mel features {@code [frames, nMels]} of {@code pcm[from, from + length)},
     * zero-padded by {@code leftPad} samples on the left and as needed on the right so every one of
     * the {@code frames} windows is in bounds.
     */
    public float[] frames(float[] pcm, int from, int length, int leftPad, int frames) {
        int nFft = spec.nFft(), hop = spec.hop(), nMels = spec.nMels();
        float[] window = spec.window(), filterbank = spec.filterbank();
        float melFloor = spec.melFloor(), logZeroGuard = spec.logZeroGuard();
        boolean power = spec.magPower() == 2f;
        if (frames == 0) return new float[0];

        float[] padded = new float[Math.max(leftPad + length, (frames - 1) * hop + nFft)];
        float preemphasis = spec.preemphasis();
        if (preemphasis > 0f) {
            // NeMo applies pre-emphasis to the raw signal before padding; x[0] is unchanged
            // and the recurrence is accumulated in double.
            if (length > 0) padded[leftPad] = pcm[from];
            for (int i = 1; i < length; i++)
                padded[leftPad + i] =
                        (float) (pcm[from + i] - preemphasis * (double) pcm[from + i - 1]);
        } else {
            System.arraycopy(pcm, from, padded, leftPad, length);
        }

        float[] output = new float[frames * nMels];
        int threads = Parallel.threads();
        float[][] fftInputs = new float[threads][nFft * 2];
        float[][] fftOutputs = new float[threads][nFft * 8];
        float[][] magnitudes = new float[threads][bins];
        Parallel.forLoop(
                0,
                frames,
                (t, slot) -> {
                    float[] fftInput = fftInputs[slot];
                    float[] fftOutput = fftOutputs[slot];
                    float[] magnitude = magnitudes[slot];
                    int offset = t * hop;
                    for (int k = 0; k < nFft; k++) fftInput[k] = window[k] * padded[offset + k];
                    fftReal(fftInput, 0, nFft, fftOutput, 0);
                    if (power) {
                        for (int b = 0; b < bins; b++) {
                            double re = fftOutput[2 * b], im = fftOutput[2 * b + 1];
                            magnitude[b] = (float) (re * re + im * im);
                        }
                    } else {
                        for (int b = 0; b < bins; b++) {
                            float p =
                                    fftOutput[2 * b] * fftOutput[2 * b]
                                            + fftOutput[2 * b + 1] * fftOutput[2 * b + 1];
                            magnitude[b] = (float) Math.sqrt(p);
                        }
                    }
                    int lastGroup = (bins - 1) & ~3;
                    for (int m = 0; m < nMels; m++) {
                        double sum = 0;
                        int base = m * bins;
                        int end = Math.min(bandEnd[m], lastGroup);
                        for (int b = bandStart[m]; b < end; b += 4)
                            sum +=
                                    magnitude[b] * filterbank[base + b]
                                            + magnitude[b + 1] * filterbank[base + b + 1]
                                            + magnitude[b + 2] * filterbank[base + b + 2]
                                            + magnitude[b + 3] * filterbank[base + b + 3];
                        for (int b = Math.max(bandStart[m], lastGroup); b < bandEnd[m]; b++)
                            sum += magnitude[b] * filterbank[base + b];
                        if (logZeroGuard > 0f) sum += logZeroGuard;
                        if (melFloor > 0f) sum = Math.max(sum, melFloor);
                        output[t * nMels + m] = (float) Math.log(sum);
                    }
                });
        return output;
    }

    /**
     * NeMo {@code per_feature} normalization, in place on frame-major features: each mel band is
     * centered and scaled by its statistics over the first {@code valid} frames (unbiased variance,
     * {@code 1e-5} added to the deviation), and frames at {@code valid} and beyond are zeroed.
     */
    public static void normalizePerFeature(float[] features, int nMels, int frames, int valid) {
        if (valid > frames)
            throw new IllegalArgumentException("valid " + valid + " > frames " + frames);
        for (int m = 0; m < nMels; m++) {
            double mean = 0;
            for (int t = 0; t < valid; t++) mean += features[t * nMels + m];
            mean = valid > 0 ? mean / valid : 0;
            double variance = 0;
            for (int t = 0; t < valid; t++) {
                double centered = features[t * nMels + m] - mean;
                variance += centered * centered;
            }
            variance = valid > 1 ? variance / (valid - 1) : 0;
            double deviation = Math.sqrt(variance) + 1e-5;
            for (int t = 0; t < valid; t++)
                features[t * nMels + m] = (float) ((features[t * nMels + m] - mean) / deviation);
            for (int t = valid; t < frames; t++) features[t * nMels + m] = 0f;
        }
    }

    /** Radix-2 real FFT; bins {@code 0..n/2} as interleaved re/im pairs. */
    void fftReal(float[] input, int inputOffset, int n, float[] output, int outputOffset) {
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
        int step = spec.nFft() / n;
        for (int k = 0; k < half; k++) {
            int index = k * step;
            float real = cos[index], imaginary = -sin[index];
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
