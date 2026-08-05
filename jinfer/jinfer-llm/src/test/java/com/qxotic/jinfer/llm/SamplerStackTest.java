package com.qxotic.jinfer.llm;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.F32FloatTensor;
import java.lang.foreign.Arena;
import java.util.Set;
import org.junit.jupiter.api.Test;

class SamplerStackTest {

    private static F32FloatTensor logits(float... values) {
        F32FloatTensor t = F32FloatTensor.allocate(Arena.ofAuto(), values.length);
        for (int i = 0; i < values.length; i++) t.setFloat(i, values[i]);
        return t;
    }

    private static Set<Integer> survivors(F32FloatTensor t) {
        var alive = new java.util.HashSet<Integer>();
        for (int i = 0; i < t.size(); i++) {
            if (t.getFloat(i) != Float.NEGATIVE_INFINITY) alive.add(i);
        }
        return alive;
    }

    @Test
    void topKKeepsTheKHighestLogits() {
        var t = logits(1f, 5f, 3f, 4f, 2f);
        Sampler.withTopK(l -> -1, 2).sampleToken(t);
        assertEquals(Set.of(1, 3), survivors(t));
    }

    @Test
    void minPMasksRelativeToTheTop() {
        // probabilities relative to max: e^0=1, e^-1~0.37, e^-3~0.05; minP 0.1 cuts the last
        var t = logits(10f, 9f, 7f);
        Sampler.withMinP(l -> -1, 0.1f).sampleToken(t);
        assertEquals(Set.of(0, 1), survivors(t));
    }

    @Test
    void nucleusKeepsTheSmallestSetCrossingTopP() {
        // even distribution over 4: each p=0.25; topP=0.6 keeps 3 (the crossing token stays)
        var t = logits(1f, 1f, 1f, 1f);
        new NucleusFilter(4, 0.6f, l -> -1).sampleToken(t);
        assertEquals(3, survivors(t).size());
    }

    @Test
    void greedyIgnoresFiltersAndIsDeterministic() {
        var t = logits(1f, 5f, 3f);
        assertEquals(1, Sampler.select(3, 0f, 2, 0.9f, 0.05f, 42).sampleToken(t));
    }

    @Test
    void fullStackSamplesOnlyFromTheFilteredSet() {
        // top-k 2 keeps ids {1,3}; the sampled token must always come from that set
        Sampler stack = Sampler.select(5, 0.8f, 2, 0.95f, 0.05f, 7);
        for (int trial = 0; trial < 50; trial++) {
            var t = logits(1f, 5f, 3f, 4f, 2f);
            int token = stack.sampleToken(t);
            assertTrue(token == 1 || token == 3, "sampled " + token);
        }
    }
}
