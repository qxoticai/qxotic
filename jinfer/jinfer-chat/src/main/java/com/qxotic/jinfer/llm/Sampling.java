package com.qxotic.jinfer.llm;

/** Fully resolved sampling settings. */
public record Sampling(float temperature, float topP, int topK, float minP, Long seed) {
    public Sampling {
        if (!(temperature >= 0)) throw new IllegalArgumentException("temperature " + temperature);
        if (!(topP > 0 && topP <= 1)) throw new IllegalArgumentException("topP " + topP);
        if (topK < 0) throw new IllegalArgumentException("topK " + topK);
        if (!(minP >= 0 && minP <= 1)) throw new IllegalArgumentException("minP " + minP);
    }

    /** Builds the {@link Sampler} for these settings; temperature 0 selects greedy decoding. */
    public Sampler sampler(int vocabularySize) {
        return Samplers.create(this, vocabularySize);
    }

    /** A copy with each non-null argument replacing the corresponding field. */
    public Sampling override(Float temperature, Float topP, Integer topK, Float minP, Long seed) {
        return new Sampling(
                temperature != null ? temperature : this.temperature,
                topP != null ? topP : this.topP,
                topK != null ? topK : this.topK,
                minP != null ? minP : this.minP,
                seed != null ? seed : this.seed);
    }
}
