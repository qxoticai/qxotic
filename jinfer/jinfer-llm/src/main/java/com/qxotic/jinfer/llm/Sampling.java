package com.qxotic.jinfer.llm;

import java.util.concurrent.ThreadLocalRandom;

/**
 * A RESOLVED sampling stack: every knob has a value, so nothing downstream needs to know where it
 * came from. Sampling is one concept and travels as one value - carrying it as five adjacent
 * same-typed fields is how a transposed {@code (temperature, topP)} pair silently changes what a
 * model produces, which no test notices and no error reports.
 *
 * <p>{@code LoadedModel.SamplingDefaults.resolve} (jinfer-chat, which is where the container's
 * opinions live) builds one of these from the layered sources: request or flag, then the GGUF's
 * {@code general.sampling.*}, then the engine baseline. Everything after that point reads values,
 * not fallbacks.
 *
 * @param temperature 0 is greedy argmax outright, and then the filters below cannot change the
 *     outcome
 * @param topP 1 disables the nucleus filter; 0 is REJECTED rather than read as "disabled" - it is
 *     almost always a temperature that landed in this slot, and {@link Sampler#select} would
 *     silently drop the filter
 * @param topK 0, or a value at or above the vocabulary size, disables the top-k filter
 * @param minP 0 disables the min-p filter
 * @param seed the RNG root, or NULL for fresh randomness on every {@link #sampler} call - which is
 *     what a server wants by default, and what a reproducible run must not have
 */
public record Sampling(float temperature, float topP, int topK, float minP, Long seed) {

    public Sampling {
        // ranges, not taste: the four floats/ints are adjacent and interchangeable to the compiler
        if (!(temperature >= 0)) { // NaN too
            throw new IllegalArgumentException("temperature " + temperature);
        }
        if (!(0 < topP && topP <= 1)) {
            throw new IllegalArgumentException("topP " + topP);
        }
        if (topK < 0) {
            throw new IllegalArgumentException("topK " + topK);
        }
        if (!(0 <= minP && minP <= 1)) {
            throw new IllegalArgumentException("minP " + minP);
        }
    }

    /**
     * The sampler stack for a model of this vocabulary size. A null {@link #seed} draws a fresh
     * root here, so two calls on the same record are independent - deliberately, since a server
     * that reused one seed would replay identical completions for identical prompts.
     */
    public Sampler sampler(int vocabularySize) {
        return Sampler.select(
                vocabularySize,
                temperature,
                topK,
                topP,
                minP,
                seed != null ? seed : ThreadLocalRandom.current().nextLong());
    }

    /**
     * This stack with the non-null arguments applied on top: one request's overrides over a
     * server's configured defaults. A null argument keeps this record's value, INCLUDING a null
     * seed, so "the request said nothing" and "the request asked for randomness" stay the same
     * thing.
     */
    public Sampling override(Float temperature, Float topP, Integer topK, Float minP, Long seed) {
        return new Sampling(
                temperature != null ? temperature : this.temperature,
                topP != null ? topP : this.topP,
                topK != null ? topK : this.topK,
                minP != null ? minP : this.minP,
                seed != null ? seed : this.seed);
    }
}
