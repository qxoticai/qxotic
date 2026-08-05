package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;

/**
 * The sampling parameters recommended for a model, resolved field-wise through a chain of opinions
 * - {@code null} where a layer has none:
 *
 * <ol>
 *   <li>the GGUF's {@code general.sampling.*} metadata (llama.cpp's convention),
 *   <li>the model author's documented recommendation, declared by the port in {@code loaded()},
 *   <li>the engine baseline (0.8, top-p 0.95 - llama.cpp's defaults).
 * </ol>
 *
 * <p>{@code Models.load} layers 1 over 2 with {@link #withFallback}; consumers finish the chain
 * with {@link #effectiveTemperature()}/{@link #effectiveTopP()} - an explicit request or
 * configuration value always wins before this record is consulted at all. Only the knobs jinfer's
 * sampler stack implements are carried; a container's {@code top_k}, penalties or mirostat settings
 * are ignored.
 */
public record SamplingDefaults(Float temperature, Float topP) {

    // the engine baseline, OK-ish for any chat model (llama.cpp's defaults)
    private static final float DEFAULT_TEMPERATURE = 0.8f;
    private static final float DEFAULT_TOP_P = 0.95f;

    /** No recommendations - every lookup falls through to the engine baseline. */
    public static final SamplingDefaults NONE = new SamplingDefaults(null, null);

    static SamplingDefaults fromGGUF(GGUF gguf) {
        return new SamplingDefaults(
                floatValue(gguf, "general.sampling.temp"),
                floatValue(gguf, "general.sampling.top_p"));
    }

    private static Float floatValue(GGUF gguf, String key) {
        return gguf.containsKey(key) ? gguf.getValue(Float.class, key) : null;
    }

    /**
     * Field-wise precedence merge: this record's values where present, {@code fallback}'s where
     * not.
     */
    SamplingDefaults withFallback(SamplingDefaults fallback) {
        return new SamplingDefaults(
                temperature != null ? temperature : fallback.temperature,
                topP != null ? topP : fallback.topP);
    }

    /** The recommended temperature, or the engine baseline when no layer has one. */
    public float effectiveTemperature() {
        return temperature != null ? temperature : DEFAULT_TEMPERATURE;
    }

    /** The recommended top-p, or the engine baseline when no layer has one. */
    public float effectiveTopP() {
        return topP != null ? topP : DEFAULT_TOP_P;
    }
}
