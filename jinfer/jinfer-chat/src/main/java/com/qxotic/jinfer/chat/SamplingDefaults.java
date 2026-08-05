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

    /** No container recommendations - every lookup falls through to the caller's fallback. */
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
}
