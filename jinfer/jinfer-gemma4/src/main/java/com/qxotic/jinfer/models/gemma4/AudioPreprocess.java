// Mel-spectrogram front-end for the Gemma 4 "gemma4a" Conformer audio encoder (E2B).
// Reverse-engineered from llama.cpp tools/mtmd/mtmd-audio.cpp + the GEMMA4A hparams in clip.cpp.
// This is the DSP the encoder-free gemma4ua path (Gemma4Audio) never needed.
//
// Pipeline (per llama.cpp): 16 kHz mono PCM -> frames of n_fft=512 at hop=160, windowed by a
// periodic Hann(320) centered in the 512 FFT buffer (96 zero-pad each side) -> real FFT ->
// power spectrum |X|^2 -> mel filterbank (Slaney/librosa scale, area-normalized) ->
// max(.,mel_floor)
// -> log10.  Output rows are [n_mel x n_frames], mel-major (out[j*nFrames + i]) matching llama.cpp.
//
// NOTE: this is the front-end only; the 24-block Conformer body (see GEMMA4A_AUDIO_ENCODER.md) is
// NOT yet ported. Verify this against llama.cpp's `log_mel_spectrogram.json` dump before relying on
// it.
package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Media;

public final class AudioPreprocess {

    public static final int SAMPLE_RATE = 16000;
    public static final int N_FFT = 512; // FFT size -> 257 bins
    public static final int WINDOW = 320; // 20 ms Hann (periodic), centered in N_FFT
    public static final int HOP = 160; // 10 ms
    static final int N_BINS = N_FFT / 2 + 1; // 257
    static final float MEL_FLOOR = 5.960464477539063e-08f; // 2^-24

    final int nMel;
    final float[] hannPadded; // [N_FFT]: Hann(320) centered, zeros elsewhere
    final float[] melFb; // [nMel * N_BINS]

    /**
     * @param nMel clip.audio.num_mel_bins from the mmproj.
     */
    public AudioPreprocess(int nMel) {
        this.nMel = nMel;
        this.hannPadded = buildHann();
        this.melFb = buildMelFilterbank(nMel);
    }

    /**
     * Downmix+resample to 16 kHz mono then compute the log-mel. Returns [nMel*nFrames], mel-major.
     */
    public float[] logMel(Media.Audio audio, int[] outFrames) {
        return logMel(toMono16k(audio), outFrames);
    }

    /** Log-mel of 16 kHz mono PCM. outFrames[0] receives the frame count. */
    public float[] logMel(float[] pcm, int[] outFrames) {
        int nFrames = Math.max(1, pcm.length / HOP + 1);
        outFrames[0] = nFrames;
        float[] out = new float[nMel * nFrames];
        float[] re = new float[N_FFT], im = new float[N_FFT];
        for (int i = 0; i < nFrames; i++) {
            int off =
                    i * HOP - N_FFT / 2
                            + HOP / 2; // center the window on the hop position (reflect at edges)
            for (int k = 0; k < N_FFT; k++) {
                int s = off + k;
                float x = (s >= 0 && s < pcm.length) ? pcm[s] : 0f;
                re[k] = x * hannPadded[k];
                im[k] = 0f;
            }
            fft(re, im); // in-place radix-2, N_FFT is a power of two
            for (int j = 0; j < nMel; j++) {
                double sum = 0.0;
                int base = j * N_BINS;
                for (int b = 0; b < N_BINS; b++) {
                    double power = (double) re[b] * re[b] + (double) im[b] * im[b];
                    sum += power * melFb[base + b];
                }
                out[j * nFrames + i] = (float) Math.log10(Math.max(sum, MEL_FLOOR));
            }
        }
        return out;
    }

    // periodic Hann(WINDOW) centered in an N_FFT buffer (matches llama.cpp centered padding)
    private static float[] buildHann() {
        float[] w = new float[N_FFT];
        int pad = (N_FFT - WINDOW) / 2; // 96
        for (int i = 0; i < WINDOW; i++) {
            w[pad + i] =
                    (float)
                            (0.5
                                    * (1.0
                                            - Math.cos(
                                                    2.0 * Math.PI * i
                                                            / WINDOW))); // periodic: /length
        }
        return w;
    }

    // Slaney/librosa mel scale (use_htk=false), Slaney area normalization (matches mtmd-audio.cpp)
    private static float[] buildMelFilterbank(int nMel) {
        final double minLogHz = 1000.0, linSlope = 3.0 / 200.0, logStep = Math.log(6.4) / 27.0;
        final double minLogMel = minLogHz * linSlope;
        java.util.function.DoubleUnaryOperator hzToMel =
                f -> (f < minLogHz) ? f * linSlope : minLogMel + Math.log(f / minLogHz) / logStep;
        java.util.function.DoubleUnaryOperator melToHz =
                m ->
                        (m < minLogMel)
                                ? m / linSlope
                                : minLogHz * Math.exp((m - minLogMel) * logStep);

        double fmin = 0.0, fmax = 0.5 * SAMPLE_RATE;
        double mLo = hzToMel.applyAsDouble(fmin), mHi = hzToMel.applyAsDouble(fmax);
        double[] hz = new double[nMel + 2];
        for (int i = 0; i < nMel + 2; i++) {
            double mel = mLo + (mHi - mLo) * ((double) i / (nMel + 1));
            hz[i] = melToHz.applyAsDouble(mel);
        }
        double binHz = (double) SAMPLE_RATE / N_FFT;
        float[] fb = new float[nMel * N_BINS];
        for (int m = 0; m < nMel; m++) {
            double fl = hz[m], fc = hz[m + 1], fr = hz[m + 2];
            double dl = Math.max(1e-30, fc - fl), dr = Math.max(1e-30, fr - fc);
            double enorm = 2.0 / Math.max(1e-30, fr - fl); // Slaney area norm
            for (int b = 0; b < N_BINS; b++) {
                double f = b * binHz;
                double w = Math.min((f - fl) / dl, (fr - f) / dr);
                if (w > 0) fb[m * N_BINS + b] = (float) (w * enorm);
            }
        }
        return fb;
    }

    // iterative radix-2 Cooley-Tukey FFT (N_FFT = 512 = 2^9), in-place
    static void fft(float[] re, float[] im) {
        int n = re.length;
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; (j & bit) != 0; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) {
                float t = re[i];
                re[i] = re[j];
                re[j] = t;
                t = im[i];
                im[i] = im[j];
                im[j] = t;
            }
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = -2.0 * Math.PI / len;
            float wR = (float) Math.cos(ang), wI = (float) Math.sin(ang);
            for (int i = 0; i < n; i += len) {
                float cR = 1f, cI = 0f;
                for (int k = 0; k < len / 2; k++) {
                    int a = i + k, b = i + k + len / 2;
                    float tR = re[b] * cR - im[b] * cI, tI = re[b] * cI + im[b] * cR;
                    re[b] = re[a] - tR;
                    im[b] = im[a] - tI;
                    re[a] += tR;
                    im[a] += tI;
                    float nR = cR * wR - cI * wI;
                    cI = cR * wI + cI * wR;
                    cR = nR;
                }
            }
        }
    }

    // downmix to mono + linear resample to 16 kHz (parity: prefer supplying 16 kHz mono via ffmpeg)
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
