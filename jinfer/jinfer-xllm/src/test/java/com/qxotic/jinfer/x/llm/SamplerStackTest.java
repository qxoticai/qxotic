package com.qxotic.jinfer.x.llm;

import static com.qxotic.jinfer.x.llm.TestLogits.*;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryView;
import java.util.HashSet;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * The sampler stack's semantics: each filter's cut, llama.cpp's chain order, composition,
 * determinism, and that sampling only ever draws from the filtered population. SIMD/scalar kernel
 * agreement is pinned separately in {@code jinfer-xkernels}.
 */
class SamplerStackTest {

    private static Sampler nucleus(int vocabularySize, float topP) {
        return Sampler.select(vocabularySize, 1f, 0, topP, 0f, 7);
    }

    private static MemoryView<?> logits(float... values) {
        MemoryView<?> t = view(values.length);
        for (int i = 0; i < values.length; i++) set(t, i, values[i]);
        return t;
    }

    private static Set<Integer> survivors(MemoryView<?> t) {
        var alive = new HashSet<Integer>();
        for (int i = 0; i < size(t); i++) {
            if (get(t, i) != Float.NEGATIVE_INFINITY) alive.add(i);
        }
        return alive;
    }

    private static Set<Integer> positive(MemoryView<?> t) {
        var alive = new HashSet<Integer>();
        for (int i = 0; i < size(t); i++) if (get(t, i) > 0) alive.add(i);
        return alive;
    }

    /** A sink inner sampler: samples nothing, leaves the masked tensor for inspection. */
    private static final Sampler SINK = l -> -1;

    @Nested
    class TopK {

        @Test
        void keepsTheKHighestLogits() {
            var t = logits(1f, 5f, 3f, 4f, 2f);
            Sampler.withTopK(SINK, 2).sampleToken(t);
            assertEquals(Set.of(1, 3), survivors(t));
        }

        @Test
        void kOneKeepsOnlyTheArgmax() {
            var t = logits(1f, 5f, 3f);
            Sampler.withTopK(SINK, 1).sampleToken(t);
            assertEquals(Set.of(1), survivors(t));
        }

        @Test
        void tiesAtTheThresholdAllSurvive() {
            var t = logits(2f, 2f, 2f, 1f);
            Sampler.withTopK(SINK, 2).sampleToken(t);
            assertEquals(Set.of(0, 1, 2), survivors(t)); // deterministic superset on ties
        }

        @Test
        void survivingLogitsAreUntouched() {
            var t = logits(1f, 5f, 3f, 4f, 2f);
            Sampler.withTopK(SINK, 2).sampleToken(t);
            assertEquals(5f, get(t, 1));
            assertEquals(4f, get(t, 3));
        }

        @Test
        void scratchReusesCleanlyAcrossTokens() {
            Sampler s = Sampler.withTopK(SINK, 2);
            s.sampleToken(logits(9f, 8f, 0f)); // hot heap from a high-logit step
            var t = logits(0.1f, 0.3f, 0.2f); // much lower logits next step
            s.sampleToken(t);
            assertEquals(Set.of(1, 2), survivors(t));
        }
    }

    @Nested
    class MinP {

        @Test
        void cutsRelativeToTheTopToken() {
            // p relative to max: e^0=1, e^-1~0.37, e^-3~0.05; minP 0.1 cuts only the last
            var t = logits(10f, 9f, 7f);
            Sampler.withMinP(SINK, 0.1f).sampleToken(t);
            assertEquals(Set.of(0, 1), survivors(t));
        }

        @Test
        void theTopTokenAlwaysSurvives() {
            var t = logits(-5f, -50f, -50f);
            Sampler.withMinP(SINK, 0.99f).sampleToken(t);
            assertTrue(survivors(t).contains(0));
        }

        @Test
        void tinyMinPKeepsEverythingFinite() {
            var t = logits(1f, 0f, -1f, Float.NEGATIVE_INFINITY);
            Sampler.withMinP(SINK, 1e-9f).sampleToken(t);
            assertEquals(Set.of(0, 1, 2), survivors(t)); // -inf stays dead
        }
    }

    @Nested
    class Nucleus {

        @Test
        void keepsTheSmallestSetCrossingTopP() {
            // even distribution over 4: each p=0.25; topP=0.6 keeps 3 (the crossing token stays)
            var t = logits(1f, 1f, 1f, 1f);
            nucleus(4, 0.6f).sampleToken(t);
            assertEquals(3, positive(t).size());
        }

        @Test
        void aDominantTokenAloneCanBeTheNucleus() {
            // p(max) ~ 1: it crosses topP=0.9 on its own
            var t = logits(20f, 0f, 0f, 0f);
            nucleus(4, 0.9f).sampleToken(t);
            assertEquals(Set.of(0), positive(t));
        }

        @Test
        void survivingNucleusValueStaysFinite() {
            var t = logits(20f, 0f, 0f, 0f);
            nucleus(4, 0.9f).sampleToken(t);
            assertTrue(get(t, 0) > 0);
        }

        @Test
        void topPNearOneKeepsEveryFiniteToken() {
            var t = logits(1f, 0.5f, 0f, -0.5f);
            nucleus(4, 0.9999f).sampleToken(t);
            assertEquals(4, positive(t).size());
        }

        @Test
        void preMaskedTokensAreNeverResurrected() {
            var t = logits(2f, Float.NEGATIVE_INFINITY, 1.9f, 1.8f);
            nucleus(4, 0.99f).sampleToken(t);
            assertTrue(!positive(t).contains(1));
        }

        @Test
        void scratchReusesCleanlyAcrossTokens() {
            Sampler s = nucleus(4, 0.6f);
            s.sampleToken(logits(9f, 0f, 0f, 0f));
            var t = logits(1f, 1f, 1f, 1f);
            s.sampleToken(t);
            assertEquals(3, positive(t).size());
        }
    }

    @Nested
    class Banning {

        @Test
        void anEmptyBanSetIsTheSameSampler() {
            assertSame(SINK, Sampler.banning(SINK, Set.of()));
        }

        @Test
        void bannedTokensAreNeverSampled() {
            Sampler s = Sampler.banning(Sampler.ARGMAX, Set.of(1));
            assertEquals(2, s.sampleToken(logits(1f, 5f, 3f))); // the argmax is banned
        }
    }

    @Nested
    class SelectDispatch {

        @Test
        void temperatureZeroIsGreedyAndIgnoresFilters() {
            var t = logits(1f, 5f, 3f);
            assertEquals(1, Sampler.select(3, 0f, 2, 0.9f, 0.05f, 42).sampleToken(t));
        }

        @Test
        void disabledKnobsFilterNothing() {
            // topK=0, topP=1, minP=0: pure temperature sampling - every token reachable
            Sampler s = Sampler.select(4, 5f, 0, 1f, 0f, 7); // hot temp flattens the draw
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 300; i++) {
                seen.add(s.sampleToken(logits(1f, 0.9f, 1.1f, 1f)));
            }
            assertEquals(Set.of(0, 1, 2, 3), seen);
        }

        @Test
        void topKAtVocabularySizeIsDisabled() {
            Sampler s = Sampler.select(3, 5f, 3, 1f, 0f, 7);
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 200; i++) seen.add(s.sampleToken(logits(1f, 1f, 1f)));
            assertEquals(Set.of(0, 1, 2), seen);
        }

        @Test
        void minPOneKeepsOnlyTheMaximum() {
            var t = logits(3f, 2f, 1f);
            Sampler.select(3, 1f, 0, 1f, 1f, 7).sampleToken(t);
            assertEquals(Set.of(0), positive(t));
        }
    }

    @Nested
    class FullStack {

        @Test
        void samplesOnlyFromTheFilteredSet() {
            // top-k 2 keeps ids {1,3}; the sampled token must always come from that set
            Sampler stack = Sampler.select(5, 0.8f, 2, 0.95f, 0.05f, 7);
            for (int trial = 0; trial < 100; trial++) {
                var t = logits(1f, 5f, 3f, 4f, 2f);
                int token = stack.sampleToken(t);
                assertTrue(token == 1 || token == 3, "sampled " + token);
            }
        }

        @Test
        void chainOrderIsTopKThenNucleusThenMinP() {
            // top-k 3 keeps {0,1,3}; over those, p ~ {0.34, 0.34, 0.31}: topP=0.5 keeps the
            // two 2f tokens (the second crosses the line at 0.69); id 3 must stay dead even
            // though the lax min-p alone would have kept it
            Sampler stack = Sampler.select(4, 1f, 3, 0.5f, 0.01f, 7);
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 300; i++) {
                seen.add(stack.sampleToken(logits(2f, 2f, 1f, 1.9f)));
            }
            assertTrue(seen.contains(0) || seen.contains(1));
            assertTrue(!seen.contains(2) && !seen.contains(3), "sampled " + seen);
        }

        @Test
        void sameSeedSameSequence() {
            Sampler a = Sampler.select(5, 1.2f, 0, 0.99f, 0f, 1234);
            Sampler b = Sampler.select(5, 1.2f, 0, 0.99f, 0f, 1234);
            for (int i = 0; i < 50; i++) {
                assertEquals(
                        a.sampleToken(logits(1f, 2f, 3f, 2f, 1f)),
                        b.sampleToken(logits(1f, 2f, 3f, 2f, 1f)));
            }
        }

        @Test
        void differentSeedsDiverge() {
            Sampler a = Sampler.select(4, 5f, 0, 1f, 0f, 1);
            Sampler b = Sampler.select(4, 5f, 0, 1f, 0f, 2);
            var sa = new StringBuilder();
            var sb = new StringBuilder();
            for (int i = 0; i < 60; i++) {
                sa.append(a.sampleToken(logits(1f, 1f, 1f, 1f)));
                sb.append(b.sampleToken(logits(1f, 1f, 1f, 1f)));
            }
            assertNotEquals(sa.toString(), sb.toString());
        }

        @Test
        void aPeakedDistributionMostlySamplesThePeak() {
            Sampler s = Sampler.select(4, 1f, 0, 1f, 0f, 99);
            int peak = 0;
            for (int i = 0; i < 200; i++) {
                if (s.sampleToken(logits(6f, 0f, 0f, 0f)) == 0) peak++;
            }
            assertTrue(peak > 190, "peak sampled " + peak + "/200"); // p(peak) ~ 0.993
        }

        @Test
        void categoricalFrequenciesTrackProbabilities() {
            // ln(2) logit gap = 2:1 odds between ids 0 and 1
            Sampler s = Sampler.select(2, 1f, 0, 1f, 0f, 5);
            int zero = 0;
            int n = 3000;
            for (int i = 0; i < n; i++) {
                if (s.sampleToken(logits((float) Math.log(2), 0f)) == 0) zero++;
            }
            double ratio = zero / (double) n;
            assertTrue(Math.abs(ratio - 2.0 / 3.0) < 0.04, "ratio " + ratio);
        }
    }
}
