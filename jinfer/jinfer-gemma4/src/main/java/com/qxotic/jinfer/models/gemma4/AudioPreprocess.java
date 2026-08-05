// Mel-spectrogram front-end for the Gemma 4 "gemma4a" Conformer audio encoder (E2B/E4B).
// Ported from llama.cpp tools/mtmd/mtmd-audio.cpp (mtmd_audio_preprocessor_gemma4a + the shared
// log_mel_spectrogram core); parity is pinned by MelParityTest against llama-mtmd-debug dumps
// (test-fixtures/audio/oracle).
//
// Recipe (exactly the reference's - each choice differs from the obvious librosa default):
//   16 kHz mono PCM -> 30 s chunks -> per chunk: semicausal pad (160 zeros left, zeros right to
//   the PyTorch frame count) -> frames of n_fft=512 at hop=160 windowed by a periodic Hann(320)
//   LEFT-ALIGNED in the 512 buffer (zeros [320..512)) -> real FFT -> MAGNITUDE |X| (not power)
//   -> HTK mel filterbank (2595*log10(1+f/700), NO Slaney area norm, fmin 0, fmax 8 kHz)
//   -> max(., 0.001) -> NATURAL log.
// Output per chunk is mel-major: data[m * frames + t], frames trimmed to the PyTorch count
//   ptFrames = (chunkLen + 160 - 321) / 160 + 1.
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Media;
import java.util.ArrayList;
import java.util.List;

public final class AudioPreprocess {

    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 512; // -> 257 bins
    public static final int WINDOW = 320; // 20 ms periodic Hann, left-aligned in the FFT frame
    public static final int HOP = 160; // 10 ms
    public static final int CHUNK_SAMPLES = 30 * SAMPLE_RATE; // model context limit per chunk
    static final int N_BINS = N_FFT / 2 + 1; // 257
    static final float MEL_FLOOR = 0.001f;

    /** One 30-second window of log-mels, time-major: {@code data[t * nMel + m]}. */
    public record MelChunk(float[] data, int frames) {}

    final int nMel;
    final float[] hann; // [N_FFT]: periodic Hann(WINDOW) in [0, WINDOW), zeros to N_FFT
    final float[] melFb; // [nMel * N_BINS]
    final int[] bandStart; // [nMel] first nonzero bin, rounded down to the 4-group boundary
    final int[] bandEnd; // [nMel] one past the last nonzero bin

    /**
     * @param nMel clip.audio.num_mel_bins from the mmproj.
     */
    public AudioPreprocess(int nMel) {
        this.nMel = nMel;
        this.hann = buildHann();
        this.melFb = buildMelFilterbank(nMel);
        this.bandStart = new int[nMel];
        this.bandEnd = new int[nMel];
        for (int m = 0; m < nMel; m++) {
            int first = N_BINS;
            int last = -1;
            for (int b = 0; b < N_BINS; b++) {
                if (melFb[m * N_BINS + b] != 0f) {
                    if (first == N_BINS) first = b;
                    last = b;
                }
            }
            // triangular filters are ~2 nonzero bins out of 257; skipping the all-zero 4-groups
            // leaves the surviving group sums bit-identical (adding a 0f product is a no-op)
            bandStart[m] = first == N_BINS ? 0 : first & ~3;
            bandEnd[m] = last + 1;
        }
    }

    /** The 16 kHz mono sample count {@link #toMono16k} would produce for {@code audio}. */
    public static int mono16kLength(Media.Audio audio) {
        int monoLen = audio.pcm().length / Math.max(1, audio.channels());
        if (audio.sampleRate() == SAMPLE_RATE) return Math.max(1, monoLen);
        return Math.max(1, (int) Math.round(monoLen * ((double) SAMPLE_RATE / audio.sampleRate())));
    }

    /** Mel frames for one chunk of {@code chunkSamples} 16 kHz samples (the PyTorch count). */
    public static int framesFor(int chunkSamples) {
        return (chunkSamples + WINDOW / 2 - (WINDOW + 1)) / HOP + 1;
    }

    /** Downmix+resample to 16 kHz mono, then log-mel per 30 s chunk. */
    public List<MelChunk> logMel(Media.Audio audio) {
        return logMel(toMono16k(audio));
    }

    /** Log-mel of 16 kHz mono PCM, one chunk per 30 s window. */
    public List<MelChunk> logMel(float[] pcm) {
        List<MelChunk> chunks = new ArrayList<>();
        for (int off = 0; off < pcm.length; off += CHUNK_SAMPLES) {
            chunks.add(chunk(pcm, off, Math.min(CHUNK_SAMPLES, pcm.length - off)));
        }
        return chunks;
    }

    private MelChunk chunk(float[] pcm, int from, int len) {
        // semicausal left pad + right pad to the PyTorch frame count (unfold(WINDOW+1, HOP) on a
        // left-padded waveform); the spectrogram then runs unpadded over this buffer
        int padLeft = WINDOW / 2; // 160
        int frames = framesFor(len);
        int paddedNeeded = (frames - 1) * HOP + N_FFT;
        int totalPad = Math.max(paddedNeeded - len, padLeft);
        float[] padded = new float[totalPad + len]; // sized so every frame's window is in range
        System.arraycopy(pcm, from, padded, padLeft, len);

        float[] out = new float[frames * nMel];
        com.qxotic.jinfer.Parallel.parallelFor(
                0,
                frames,
                t -> {
                    // per-frame scratch: the FFT is the reference's op-for-op port, so the
                    // per-frame numbers are identical no matter which thread runs the frame
                    float[] fftIn = new float[N_FFT * 2];
                    float[] fftOut = new float[N_FFT * 8];
                    float[] mag = new float[N_BINS];
                    int off = t * HOP;
                    for (int k = 0; k < WINDOW; k++) { // hann[WINDOW..N_FFT) is all zeros
                        fftIn[k] = hann[k] * padded[off + k];
                    }
                    fftReal(fftIn, 0, N_FFT, fftOut, 0);
                    for (int b = 0; b < N_BINS; b++) {
                        float power =
                                fftOut[2 * b] * fftOut[2 * b]
                                        + fftOut[2 * b + 1] * fftOut[2 * b + 1];
                        mag[b] = (float) Math.sqrt(power);
                    }
                    int lastGroup = (N_BINS - 1) & ~3; // the reference's scalar-tail start (256)
                    for (int m = 0; m < nMel; m++) {
                        // the reference's 4-unrolled accumulation restricted to the filter's
                        // nonzero band: whole groups only (zero products inside a surviving
                        // group are float no-ops), so every group sum is bit-identical
                        double sum = 0.0;
                        int base = m * N_BINS;
                        int end = Math.min(bandEnd[m], lastGroup);
                        for (int b = bandStart[m]; b < end; b += 4) {
                            sum +=
                                    mag[b] * melFb[base + b]
                                            + mag[b + 1] * melFb[base + b + 1]
                                            + mag[b + 2] * melFb[base + b + 2]
                                            + mag[b + 3] * melFb[base + b + 3];
                        }
                        for (int b = Math.max(bandStart[m], lastGroup); b < bandEnd[m]; b++) {
                            sum += mag[b] * melFb[base + b];
                        }
                        out[t * nMel + m] = (float) Math.log(Math.max(sum, MEL_FLOOR));
                    }
                });
        return new MelChunk(out, frames);
    }

    // periodic Hann(WINDOW) left-aligned in an N_FFT buffer (the reference does NOT center it).
    // Float op-for-op with the reference (0.5f - 0.5f*cosf((2f*pi*i)/len)): double-computed
    // coefficients shift high-frequency leakage bins ~1% relative, visible in ln-space parity.
    private static float[] buildHann() {
        float[] w = new float[N_FFT];
        float pi = (float) Math.PI;
        for (int i = 0; i < WINDOW; i++) {
            float arg = (2.0f * pi * i) / WINDOW;
            w[i] = 0.5f - 0.5f * (float) Math.cos(arg);
        }
        return w;
    }

    // HTK mel scale, triangular filters, NO area normalization (gemma4a's fill_mel_filterbank)
    private static float[] buildMelFilterbank(int nMel) {
        double fmax = 0.5 * SAMPLE_RATE;
        double mLo = hzToMel(0.0), mHi = hzToMel(fmax);
        double[] hz = new double[nMel + 2];
        for (int i = 0; i < nMel + 2; i++) {
            hz[i] = melToHz(mLo + (mHi - mLo) * ((double) i / (nMel + 1)));
        }
        double binHz = (double) SAMPLE_RATE / N_FFT;
        float[] fb = new float[nMel * N_BINS];
        for (int m = 0; m < nMel; m++) {
            double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
            double dl = Math.max(1e-30, fc - fl), dr = Math.max(1e-30, fr - fc);
            for (int b = 0; b < N_BINS; b++) {
                double f = b * binHz;
                double w = 0.0;
                if (f >= fl && f <= fc) {
                    w = (f - fl) / dl;
                } else if (f > fc && f <= fr) {
                    w = (fr - f) / dr;
                }
                fb[m * N_BINS + b] = (float) w;
            }
        }
        return fb;
    }

    private static double hzToMel(double hz) {
        return 2595.0 * Math.log10(1.0 + hz / 700.0);
    }

    private static double melToHz(double mel) {
        return 700.0 * (Math.pow(10.0, mel / 2595.0) - 1.0);
    }

    // llama.cpp's recursive real-input radix-2 FFT (mtmd-audio.cpp fft_impl), ported with the
    // same table twiddles, scratch layout and operation order so near-zero leakage bins round
    // identically - an algebraically equal FFT with different op order fails ln-space parity.
    private static final float[] SIN_VALS = new float[N_FFT];
    private static final float[] COS_VALS = new float[N_FFT];

    static {
        for (int i = 0; i < N_FFT; i++) {
            double theta = (2.0 * Math.PI * i) / N_FFT;
            SIN_VALS[i] = (float) Math.sin((double) (float) theta);
            COS_VALS[i] = (float) Math.cos((double) (float) theta);
        }
    }

    /**
     * Real-input forward FFT: {@code in} holds N reals (capacity 2N, tail is scratch); {@code out}
     * holds interleaved re/im (capacity 8N, tail is scratch).
     */
    static void fftReal(float[] in, int inOff, int n, float[] out, int outOff) {
        if (n == 1) {
            out[outOff] = in[inOff];
            out[outOff + 1] = 0.0f;
            return;
        }
        int half = n / 2;
        // even reals into the input buffer's tail, recurse, then odds reuse the same scratch
        int evenOff = inOff + n;
        for (int i = 0; i < half; i++) {
            in[evenOff + i] = in[inOff + 2 * i];
        }
        int evenFft = outOff + 2 * n;
        fftReal(in, evenOff, half, out, evenFft);
        for (int i = 0; i < half; i++) {
            in[evenOff + i] = in[inOff + 2 * i + 1];
        }
        int oddFft = evenFft + n;
        fftReal(in, evenOff, half, out, oddFft);

        int step = N_FFT / n;
        for (int k = 0; k < half; k++) {
            int idx = k * step; // t = 2*pi*k/n
            float re = COS_VALS[idx];
            float im = -SIN_VALS[idx];
            float reOdd = out[oddFft + 2 * k];
            float imOdd = out[oddFft + 2 * k + 1];
            out[outOff + 2 * k] = out[evenFft + 2 * k] + re * reOdd - im * imOdd;
            out[outOff + 2 * k + 1] = out[evenFft + 2 * k + 1] + re * imOdd + im * reOdd;
            out[outOff + 2 * (k + half)] = out[evenFft + 2 * k] - re * reOdd + im * imOdd;
            out[outOff + 2 * (k + half) + 1] = out[evenFft + 2 * k + 1] - re * imOdd - im * reOdd;
        }
    }

    // downmix to mono + linear resample to 16 kHz (parity: prefer supplying 16 kHz mono)
    static float[] toMono16k(Media.Audio audio) {
        int ch = Math.max(1, audio.channels());
        float[] in = audio.pcm();
        int frames = in.length / ch;
        float[] mono = ch == 1 ? in : new float[frames];
        if (ch != 1)
            for (int i = 0; i < frames; i++) {
                float s = 0f;
                for (int c = 0; c < ch; c++) s += in[i * ch + c];
                mono[i] = s / ch;
            }
        if (audio.sampleRate() == SAMPLE_RATE) return mono;
        double ratio = (double) SAMPLE_RATE / audio.sampleRate();
        float[] outp = new float[Math.max(1, (int) Math.round(mono.length * ratio))];
        for (int i = 0; i < outp.length; i++) {
            double sp = i / ratio;
            int j = (int) sp;
            double fr = sp - j;
            float a = mono[Math.min(j, mono.length - 1)],
                    b = mono[Math.min(j + 1, mono.length - 1)];
            outp[i] = (float) (a + (b - a) * fr);
        }
        return outp;
    }
}
