package com.llm4j.model.llama;

public final class RoPE {

    public static float[][] precomputeFreqsCis(int contextLength, int headSize, double theta) {
        return precomputeFreqsCis(contextLength, headSize, theta,
                false, 0, 0, 0, 0);
    }

    public static float[][] precomputeFreqsCis(int contextLength, int headSize, double theta, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        return precomputeFreqsCis(contextLength, headSize, theta,
                true, scaleFactor, loFreqFactor, hiFreqFactor, oldContextLength);
    }

    public static float[][] precomputeFreqsCis(int contextLength, int headSize, double theta, float[] ropeFreqs) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize)) / ropeFreqs[i / 2];
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new float[][]{cr, ci};
    }

    private static float[][] precomputeFreqsCis(int contextLength, int headSize, double theta,
                                                boolean ropeScaling, float scaleFactor, float loFreqFactor, float hiFreqFactor, float oldContextLength) {
        assert headSize % 2 == 0;
        float[] cr = new float[contextLength * (headSize / 2)];
        float[] ci = new float[contextLength * (headSize / 2)];
        int n = 0;
        for (int pos = 0; pos < contextLength; ++pos) {
            for (int i = 0; i < headSize; i += 2) {
                float freq = (float) (1.0 / Math.pow(theta, i / (double) headSize));
                if (ropeScaling) {
                    // Llama 3.1+ scaling
                    float loFreqWavelen = oldContextLength / loFreqFactor;
                    float hiFreqWavelen = oldContextLength / hiFreqFactor;
                    float wavelen = (float) (2.0 * Math.PI / freq);
                    if (wavelen < hiFreqWavelen) {
                        freq = freq;
                    } else if (wavelen > loFreqWavelen) {
                        freq = freq / scaleFactor;
                    } else {
                        float smooth = (oldContextLength / wavelen - loFreqFactor) / (hiFreqFactor - loFreqFactor);
                        freq = (1.0f - smooth) * freq / scaleFactor + smooth * freq;
                    }
                }
                float val = pos * freq;
                cr[n] = (float) Math.cos(val);
                ci[n] = (float) Math.sin(val);
                n++;
            }
        }
        assert contextLength * (headSize / 2) == n;
        return new float[][]{cr, ci};
    }
}
