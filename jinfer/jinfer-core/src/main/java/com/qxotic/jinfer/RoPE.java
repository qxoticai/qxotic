package com.qxotic.jinfer;

/**
 * Precomputed rotary position embedding tables: plain RoPE, llama3 per-frequency scaling ({@code
 * rope_freqs.weight}) and YaRN. One interleaved per-pair cos/sin layout serves both application
 * styles - interleaved ({@code ROPE_TYPE_NORM}) and rotate-half ({@code NEOX}) - so a port picks a
 * table, not a layout.
 */
public final class RoPE {
    /**
     * Regular (GPT-J / interleaved) RoPE over the first {@code 2*ropeHalf} dims of one head: pairs
     * adjacent dims (2i, 2i+1). This is the GGUF "llama" rope convention (ROPE_TYPE_NORM); the
     * weights are permuted at conversion so the interleaved rotation reproduces HF rotate-half.
     * {@code cr}/{@code ci} are a {@link #precomputeFreqsCis}-family table at stride {@code
     * ropeHalf}.
     */
    /** Precomputed rotary tables: cos and sin per (position, frequency). */
    public record Freqs(float[] cos, float[] sin) {}

    /**
     * Fills rows for positions {@code [fromPosition, fromPosition + count)} into a caller's buffers
     * at {@code row * halfHead}, where {@code row} is the position's index WITHIN the range.
     *
     * <p>This is how rotary values reach the model: a state fills its scratch once per ingest and
     * every layer reads it, because the values depend only on the position and the schedule, never
     * on the layer. Nothing is retained between ingests, so no buffer anywhere is sized by context
     * - which for a sliding-window model like Gemma 4 is the difference between 1.5 MB of flat
     * scratch and 402 MB of table at 128k, against a KV cache of 811 MB.
     *
     * <p>The arithmetic is character-for-character that of {@link #precomputeFreqsCis}, so a range
     * holds exactly the values the same positions held in a front-loaded table.
     */
    public static void fill(
            FloatTensor cos,
            FloatTensor sin,
            int fromPosition,
            int count,
            int headSize,
            double theta) {
        assert headSize % 2 == 0;
        int halfHead = headSize / 2;
        long n = 0;
        for (int p = 0; p < count; p++) {
            int pos = fromPosition + p;
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float val = pos * freq;
                cos.setFloat(n, (float) Math.cos(val));
                sin.setFloat(n, (float) Math.sin(val));
                n++;
            }
        }
        assert n == (long) count * halfHead;
    }

    /** {@link #fill} with llama3 per-frequency scaling ({@code rope_freqs.weight}). */
    public static void fillFromFreqs(
            FloatTensor cos,
            FloatTensor sin,
            int fromPosition,
            int count,
            int headSize,
            double ropeTheta,
            float[] ropeFreqFactors) {
        int halfHead = ropeFreqFactors.length;
        assert halfHead == headSize / 2;
        long n = 0;
        for (int p = 0; p < count; p++) {
            int pos = fromPosition + p;
            for (int i = 0; i < halfHead; i++) {
                float baseFreq = (float) (1.0 / Math.pow(ropeTheta, (2.0 * i) / headSize));
                float val = pos * baseFreq / ropeFreqFactors[i];
                cos.setFloat(n, (float) Math.cos(val));
                sin.setFloat(n, (float) Math.sin(val));
                n++;
            }
        }
        assert n == (long) count * halfHead;
    }

    /**
     * {@link #fill} with YaRN interpolation. The schedule is position-independent, so it is
     * computed once per call exactly as the front-loaded table computed it once per model.
     */
    public static void fillYarn(
            FloatTensor cos,
            FloatTensor sin,
            int fromPosition,
            int count,
            int headSize,
            double ropeTheta,
            float ropeScalingFactor,
            int originalContextLength,
            float betaFast,
            float betaSlow,
            float extFactor,
            float attnFactor) {
        assert headSize % 2 == 0;
        int halfHead = headSize / 2;
        float freqScale = ropeScalingFactor == 0f ? 1f : 1f / ropeScalingFactor;
        float corrStart =
                Math.max(
                        0f,
                        (float)
                                Math.floor(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaFast,
                                                (float) ropeTheta)));
        float corrEnd =
                Math.min(
                        headSize - 1f,
                        (float)
                                Math.ceil(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaSlow,
                                                (float) ropeTheta)));
        float mscale =
                attnFactor
                        * (extFactor != 0f
                                ? (float) (1.0 + 0.1 * Math.log(1.0 / Math.max(1e-12, freqScale)))
                                : 1f);
        long n = 0;
        for (int p = 0; p < count; p++) {
            int pos = fromPosition + p;
            for (int i = 0; i < headSize; i += 2) {
                double baseFreq = 1.0 / Math.pow(ropeTheta, i / (double) headSize);
                float thetaExtrap = (float) (pos * baseFreq);
                float thetaInterp = freqScale * thetaExtrap;
                float ramp = yarnRamp(corrStart, corrEnd, i) * extFactor;
                float theta = thetaInterp * (1f - ramp) + thetaExtrap * ramp;
                cos.setFloat(n, (float) (Math.cos(theta) * mscale));
                sin.setFloat(n, (float) (Math.sin(theta) * mscale));
                n++;
            }
        }
        assert n == (long) count * halfHead;
    }

    public static Freqs precomputeFreqsCis(int contextLength, int headSize, double theta) {
        assert headSize % 2 == 0;
        int halfHead = headSize / 2;
        float[] cr = new float[contextLength * halfHead];
        float[] ci = new float[contextLength * halfHead];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * halfHead == n;
        return new Freqs(cr, ci);
    }

    public static Freqs precomputeFreqsCisFromFreqs(
            int contextLength, int headSize, double ropeTheta, float[] ropeFreqFactors) {
        // freq_factors are divisors on top of the standard RoPE base frequencies:
        // theta_i = pos * (1 / (ropeTheta^(2i/headSize))) / freqFactors[i]
        int halfHead = ropeFreqFactors.length;
        assert halfHead == headSize / 2;
        float[] cr = new float[contextLength * halfHead];
        float[] ci = new float[contextLength * halfHead];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < halfHead; i++) {
                float baseFreq = (float) (1.0 / Math.pow(ropeTheta, (2.0 * i) / headSize));
                float val = pos * baseFreq / ropeFreqFactors[i];
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * halfHead == n;
        return new Freqs(cr, ci);
    }

    static double yarnCorrDim(int nDims, int nCtxOrig, float nRot, float base) {
        return nDims * Math.log(nCtxOrig / (nRot * 2.0 * Math.PI)) / (2.0 * Math.log(base));
    }

    static float yarnRamp(float low, float high, int i0) {
        float y = (i0 / 2f - low) / Math.max(0.001f, high - low);
        return 1f - Math.min(1f, Math.max(0f, y));
    }

    /**
     * YaRN-scaled RoPE tables (cos/sin) in the same interleaved per-pair layout as {@link
     * #precomputeFreqsCis} — the attention mscale is baked into cos/sin. Mirrors ggml's {@code
     * rope_yarn} (theta interpolated/extrapolated by the correction ramp; in-kernel mscale = 1 +
     * 0.1·ln(1/freq_scale) when ext_factor != 0), times {@code attnFactor} — llama.cpp's {@code
     * cparams.yarn_attn_factor}, the extra magnitude factor it folds onto the kernel mscale. Pass
     * 1.0 for the plain YaRN magnitude (gpt-oss); pass 1/(kernel mscale) to net a magnitude of 1.0
     * (mistral3, whose yarn_log_multiplier cancels the kernel mscale).
     */
    public static Freqs precomputeFreqsCisYarn(
            int contextLength,
            int headSize,
            double ropeTheta,
            float ropeScalingFactor,
            int originalContextLength,
            float betaFast,
            float betaSlow,
            float extFactor,
            float attnFactor) {
        assert headSize % 2 == 0;
        int halfHead = headSize / 2;
        float[] cr = new float[contextLength * halfHead];
        float[] ci = new float[contextLength * halfHead];
        float freqScale = ropeScalingFactor == 0f ? 1f : 1f / ropeScalingFactor;
        float corrStart =
                Math.max(
                        0f,
                        (float)
                                Math.floor(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaFast,
                                                (float) ropeTheta)));
        float corrEnd =
                Math.min(
                        headSize - 1f,
                        (float)
                                Math.ceil(
                                        yarnCorrDim(
                                                headSize,
                                                originalContextLength,
                                                betaSlow,
                                                (float) ropeTheta)));
        float mscale =
                attnFactor
                        * (extFactor != 0f
                                ? (float) (1.0 + 0.1 * Math.log(1.0 / Math.max(1e-12, freqScale)))
                                : 1f);
        int n = 0;
        for (int pos = 0; pos < contextLength; pos++) {
            for (int i = 0; i < headSize; i += 2) {
                double baseFreq = 1.0 / Math.pow(ropeTheta, i / (double) headSize);
                float thetaExtrap = (float) (pos * baseFreq);
                float thetaInterp = freqScale * thetaExtrap;
                float ramp = yarnRamp(corrStart, corrEnd, i) * extFactor;
                float theta = thetaInterp * (1f - ramp) + thetaExtrap * ramp;
                cr[n] = (float) (Math.cos(theta) * mscale);
                ci[n] = (float) (Math.sin(theta) * mscale);
                n++;
            }
        }
        assert contextLength * halfHead == n;
        return new Freqs(cr, ci);
    }
}
