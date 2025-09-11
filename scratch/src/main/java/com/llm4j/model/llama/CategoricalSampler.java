package com.llm4j.model.llama;

import com.llm4j.span.FloatSpan;

import java.util.random.RandomGenerator;

record CategoricalSampler(RandomGenerator rng) implements Sampler2 {

    @Override
    public int applyAsInt(FloatSpan logits) {

        var logitsOps = Util.directAccess(logits);

        // sample index from probabilities (they must sum to 1!)
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logitsOps.getElementAt(logits, i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return (int) logits.size() - 1; // in case of rounding errors
    }
}
