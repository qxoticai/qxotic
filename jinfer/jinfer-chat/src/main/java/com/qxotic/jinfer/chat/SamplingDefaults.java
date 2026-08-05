package com.qxotic.jinfer.chat;

import com.qxotic.format.gguf.GGUF;

/**
 * The sampling parameters a GGUF recommends for its model ({@code general.sampling.*} metadata,
 * llama.cpp's convention) - {@code null} where the container carries none. Consumers use these as
 * the LAST fallback: an explicit request or configuration value always wins, these win over the
 * engine's hardcoded defaults. Only the knobs jinfer's sampler stack implements are read; a
 * container's {@code top_k}, penalties or mirostat settings are ignored.
 */
public record SamplingDefaults(Float temperature, Float topP) {

    /**
     * The engine baseline, OK-ish for any chat model (llama.cpp's defaults): the bottom of the
     * three-layer chain {@code request/config > container recommendation > this baseline}. Use
     * {@link #effectiveTemperature()}/{@link #effectiveTopP()} to resolve the full chain's tail.
     */
    public static final float DEFAULT_TEMPERATURE = 0.8f;

    public static final float DEFAULT_TOP_P = 0.95f;

    /** No container recommendations - every lookup falls through to the engine baseline. */
    public static final SamplingDefaults NONE = new SamplingDefaults(null, null);

    public static SamplingDefaults fromGGUF(GGUF gguf) {
        return new SamplingDefaults(
                floatValue(gguf, "general.sampling.temp"),
                floatValue(gguf, "general.sampling.top_p"));
    }

    private static Float floatValue(GGUF gguf, String key) {
        return gguf.containsKey(key) ? gguf.getValue(Float.class, key) : null;
    }

    public float temperatureOr(float fallback) {
        return temperature != null ? temperature : fallback;
    }

    public float topPOr(float fallback) {
        return topP != null ? topP : fallback;
    }

    /**
     * Field-wise precedence merge: this record's values where present, {@code fallback}'s where
     * not. {@code Models.load} uses it to layer the GGUF's recommendations over the port's
     * (model-author) recommendations.
     */
    public SamplingDefaults withFallback(SamplingDefaults fallback) {
        return new SamplingDefaults(
                temperature != null ? temperature : fallback.temperature,
                topP != null ? topP : fallback.topP);
    }

    /** The container's recommendation, or the engine baseline when it carries none. */
    public float effectiveTemperature() {
        return temperatureOr(DEFAULT_TEMPERATURE);
    }

    /** The container's recommendation, or the engine baseline when it carries none. */
    public float effectiveTopP() {
        return topPOr(DEFAULT_TOP_P);
    }
}
