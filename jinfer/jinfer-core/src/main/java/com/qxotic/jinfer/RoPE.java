// Rotary position embedding tables, shared across architectures. Each method precomputes the
// per-(position, pair) cos/sin tables a model's attention applies; the three variants cover plain
// RoPE, "llama3" per-frequency scaling (rope_freqs.weight), and YaRN scaling. The tables use one
// interleaved per-pair layout consumed by both interleaved (ROPE_TYPE_NORM) and rotate-half (NEOX)
// attention — only the application pairing differs, not the table.
package com.qxotic.jinfer;

public final class RoPE {
    /**
     * Regular (GPT-J / interleaved) RoPE over the first {@code 2*ropeHalf} dims of one head: pairs
     * adjacent dims (2i, 2i+1). This is the GGUF "llama" rope convention (ROPE_TYPE_NORM); the
     * weights are permuted at conversion so the interleaved rotation reproduces HF rotate-half.
     * {@code cr}/{@code ci} are a {@link #precomputeFreqsCis}-family table at stride {@code
     * ropeHalf}.
     */
    public static void applyInterleaved(
            FloatTensor q, int headOffset, int position, float[] cr, float[] ci, int ropeHalf) {
        int base = position * ropeHalf;
        for (int i = 0; i < ropeHalf; i++) {
            float fcr = cr[base + i];
            float fci = ci[base + i];
            int idx = headOffset + 2 * i;
            float v0 = q.getFloat(idx);
            float v1 = q.getFloat(idx + 1);
            q.setFloat(idx, v0 * fcr - v1 * fci);
            q.setFloat(idx + 1, v0 * fci + v1 * fcr);
        }
    }

    /**
     * Rotate-half (NEOX) RoPE over one head: pairs dim {@code i} with {@code i+ropeHalf} (the
     * {@code (i, i+ropeHalf)} layout HF/gpt-oss apply directly, no conversion-time permutation).
     * Same cos/sin table layout as {@link #applyInterleaved} (stride {@code ropeHalf}).
     */
    public static void applyNeox(
            FloatTensor q, long headOffset, int position, float[] cr, float[] ci, int ropeHalf) {
        int base = position * ropeHalf;
        for (int i = 0; i < ropeHalf; i++) {
            float fcr = cr[base + i];
            float fci = ci[base + i];
            float v0 = q.getFloat(headOffset + i);
            float v1 = q.getFloat(headOffset + i + ropeHalf);
            q.setFloat(headOffset + i, v0 * fcr - v1 * fci);
            q.setFloat(headOffset + i + ropeHalf, v0 * fci + v1 * fcr);
        }
    }

    /**
     * Rotate-half (NEOX) RoPE over {@code nHeads} consecutive heads of {@code headSize} each, with
     * the cos/sin table held in native F32 tensors (the layout Llama/Gemma keep). Bit-identical to
     * {@link #applyNeox(FloatTensor, int, int, float[], float[], int)} per head.
     */
    public static void applyNeox(
            FloatTensor tensor,
            long offset,
            int nHeads,
            int headSize,
            int halfHead,
            int position,
            F32FloatTensor cr,
            F32FloatTensor ci) {
        for (int h = 0; h < nHeads; h++) {
            long poffset = offset + (long) h * headSize;
            for (int i = 0; i < halfHead; i++) {
                float fcr = cr.getFloat(position * halfHead + i);
                float fci = ci.getFloat(position * halfHead + i);
                float v0 = tensor.getFloat(poffset + i);
                float v1 = tensor.getFloat(poffset + i + halfHead);
                tensor.setFloat(poffset + i, v0 * fcr - v1 * fci);
                tensor.setFloat(poffset + i + halfHead, v0 * fci + v1 * fcr);
            }
        }
    }

    public static Pair<float[], float[]> precomputeFreqsCis(
            int contextLength, int headSize, double theta) {
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
        return new Pair<>(cr, ci);
    }

    public static Pair<float[], float[]> precomputeFreqsCisFromFreqs(
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
        return new Pair<>(cr, ci);
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
    public static Pair<float[], float[]> precomputeFreqsCisYarn(
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
        return new Pair<>(cr, ci);
    }
}
