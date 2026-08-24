package com.qxotic.jinfer.llm;

import static com.qxotic.jinfer.llm.TestLogits.*;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * The sampler stack's semantics: each filter's cut, llama.cpp's chain order, composition,
 * determinism, and that sampling only ever draws from the filtered population. SIMD/scalar kernel
 * agreement is pinned separately in {@code jinfer-kernels}.
 */
class SamplerStackTest {
    private static Sampler sampler(
            int vocabularySize, float temperature, int topK, float topP, float minP, long seed) {
        return new Sampling(temperature, topP, topK, minP, seed).sampler(vocabularySize);
    }

    private static MemoryView<MemorySegment> logits(float... values) {
        MemoryView<MemorySegment> t = view(values.length);
        for (int i = 0; i < values.length; i++) set(t, i, values[i]);
        return t;
    }

    private static Set<Integer> positiveIds(MemoryView<?> t) {
        var alive = new HashSet<Integer>();
        for (int i = 0; i < size(t); i++) if (get(t, i) > 0) alive.add(i);
        return alive;
    }

    @Nested
    class TopK {

        @Test
        void keepsTheKHighestLogits() {
            int[] tokenIds = new int[2];
            float[] values = new float[2];

            Samplers.selectTopK(logits(1f, 5f, 3f, 4f, 2f), tokenIds, values);

            assertArrayEquals(new int[] {1, 3}, tokenIds);
            assertArrayEquals(new float[] {5f, 4f}, values);
        }

        @Test
        void kOneKeepsOnlyTheArgmax() {
            int[] tokenIds = new int[1];
            float[] values = new float[1];

            Samplers.selectTopK(logits(1f, 5f, 3f), tokenIds, values);

            assertArrayEquals(new int[] {1}, tokenIds);
            assertArrayEquals(new float[] {5f}, values);
        }

        @Test
        void tiesKeepExactlyKLowestTokenIds() {
            int[] tokenIds = new int[2];
            float[] values = new float[2];

            Samplers.selectTopK(logits(2f, 2f, 2f, 1f), tokenIds, values);

            assertArrayEquals(new int[] {0, 1}, tokenIds);
            assertArrayEquals(new float[] {2f, 2f}, values);
        }

        @Test
        void selectionDoesNotModifyDenseLogits() {
            var t = logits(1f, 5f, 3f, 4f, 2f);
            int[] tokenIds = new int[2];
            float[] values = new float[2];

            Samplers.selectTopK(t, tokenIds, values);

            for (int i = 0; i < 5; i++) assertEquals(new float[] {1, 5, 3, 4, 2}[i], get(t, i));
        }

        @Test
        void scratchReusesCleanlyAcrossTokens() {
            int[] tokenIds = new int[2];
            float[] values = new float[2];
            Samplers.selectTopK(logits(9f, 8f, 0f), tokenIds, values);

            Samplers.selectTopK(logits(0.1f, 0.3f, 0.2f), tokenIds, values);

            assertArrayEquals(new int[] {1, 2}, tokenIds);
            assertArrayEquals(new float[] {0.3f, 0.2f}, values);
        }

        @Test
        void matchesAFullSort() {
            Random random = new Random(7);
            for (int size = 1; size <= 64; size++) {
                float[] source = new float[size];
                Integer[] expected = new Integer[size];
                for (int token = 0; token < size; token++) {
                    source[token] = random.nextInt(11) - 5;
                    expected[token] = token;
                }
                Arrays.sort(
                        expected,
                        Comparator.<Integer>comparingDouble(token -> source[token])
                                .reversed()
                                .thenComparingInt(Integer::intValue));

                for (int k = 1; k <= size; k++) {
                    int[] tokenIds = new int[k];
                    float[] values = new float[k];
                    Samplers.selectTopK(logits(source), tokenIds, values);

                    for (int i = 0; i < k; i++) {
                        assertEquals(expected[i], tokenIds[i]);
                        assertEquals(source[expected[i]], values[i]);
                    }
                }
            }
        }
    }

    @Nested
    class MinP {

        @Test
        void cutsRelativeToTheTopToken() {
            // p relative to max: e^0=1, e^-1~0.37, e^-3~0.05; minP 0.1 cuts only the last
            assertEquals(
                    2, Samplers.retainMinP(new float[] {10f, 9f, 7f}, 3, (float) Math.log(0.1f)));
        }

        @Test
        void theTopTokenAlwaysSurvives() {
            assertEquals(
                    1,
                    Samplers.retainMinP(new float[] {-5f, -50f, -50f}, 3, (float) Math.log(0.99f)));
        }

        @Test
        void tinyMinPKeepsEverythingFinite() {
            assertEquals(
                    3,
                    Samplers.retainMinP(
                            new float[] {1f, 0f, -1f, Float.NEGATIVE_INFINITY},
                            4,
                            (float) Math.log(1e-9f)));
        }

        @Test
        void aCandidateExactlyAtTheThresholdSurvives() {
            assertEquals(
                    2,
                    Samplers.retainMinP(
                            new float[] {0f, (float) Math.log(0.5)}, 2, (float) Math.log(0.5)));
        }
    }

    @Nested
    class Nucleus {

        @Test
        void keepsTheSmallestSetCrossingTopP() {
            // even distribution over 4: each p=0.25; topP=0.6 keeps 3 (the crossing token stays)
            assertEquals(3, Samplers.retainTopP(new float[] {1f, 1f, 1f, 1f}, 4, 0.6f));
        }

        @Test
        void filteringDoesNotRewriteLogits() {
            float[] values = {20f, 0f, 0f, 0f};
            Samplers.retainTopP(values, values.length, 0.9f);
            assertArrayEquals(new float[] {20f, 0f, 0f, 0f}, values);
        }

        @Test
        void preMaskedTokensAreNeverResurrected() {
            assertEquals(
                    3,
                    Samplers.retainTopP(
                            new float[] {2f, 1.9f, 1.8f, Float.NEGATIVE_INFINITY}, 4, 0.99f));
        }

        @Test
        void exactTargetKeepsTheCrossingCandidate() {
            assertEquals(1, Samplers.retainTopP(new float[] {0f, 0f}, 2, 0.5f));
        }

        @Test
        void densePathAlwaysKeepsTheTopToken() {
            var values = logits(2f, 1f, 0f);
            sampler(3, 1f, 0, Float.MIN_VALUE, 0f, 7).sampleToken(values);
            assertEquals(Set.of(0), positiveIds(values));
        }

        @Test
        void densePathMatchesAFullSort() {
            Random random = new Random(11);
            for (int size = 1; size <= 64; size++) {
                float[] source = new float[size];
                Integer[] order = new Integer[size];
                for (int token = 0; token < size; token++) {
                    source[token] = 5f * token / size;
                    order[token] = token;
                }
                for (int i = size - 1; i > 0; i--) {
                    int j = random.nextInt(i + 1);
                    float swap = source[i];
                    source[i] = source[j];
                    source[j] = swap;
                }
                Arrays.sort(
                        order,
                        Comparator.<Integer>comparingDouble(token -> source[token]).reversed());
                float topP = random.nextFloat(0.05f, 1f);

                double denominator = 0;
                for (float value : source) denominator += Math.exp(value - source[order[0]]);
                double cumulative = 0;
                Set<Integer> expected = new HashSet<>();
                for (int token : order) {
                    expected.add(token);
                    cumulative += Math.exp(source[token] - source[order[0]]);
                    if (cumulative >= topP * denominator) break;
                }

                var values = logits(source);
                sampler(size, 1f, 0, topP, 0f, 7).sampleToken(values);
                assertEquals(expected, positiveIds(values), "vocabulary size " + size);
            }
        }
    }

    @Nested
    class Banning {

        @Test
        void anEmptyBanSetIsTheSameSampler() {
            assertSame(Sampler.ARGMAX, Sampler.banning(Sampler.ARGMAX, Set.of()));
        }

        @Test
        void bannedTokensAreNeverSampled() {
            Sampler s = Sampler.banning(Sampler.ARGMAX, Set.of(1));
            assertEquals(2, s.sampleToken(logits(1f, 5f, 3f))); // the argmax is banned
        }
    }

    @Nested
    class Creation {

        @Test
        void temperatureZeroIsGreedyAndIgnoresFilters() {
            var t = logits(1f, 5f, 3f);
            assertEquals(1, sampler(3, 0f, 2, 0.9f, 0.05f, 42).sampleToken(t));
        }

        @Test
        void disabledKnobsFilterNothing() {
            // topK=0, topP=1, minP=0: pure temperature sampling - every token reachable
            Sampler s = sampler(4, 5f, 0, 1f, 0f, 7); // hot temp flattens the draw
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 300; i++) {
                seen.add(s.sampleToken(logits(1f, 0.9f, 1.1f, 1f)));
            }
            assertEquals(Set.of(0, 1, 2, 3), seen);
        }

        @Test
        void topKAtVocabularySizeIsDisabled() {
            Sampler s = sampler(3, 5f, 3, 1f, 0f, 7);
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 200; i++) seen.add(s.sampleToken(logits(1f, 1f, 1f)));
            assertEquals(Set.of(0, 1, 2), seen);
        }

        @Test
        void minPOneKeepsOnlyTheMaximum() {
            var t = logits(3f, 2f, 1f);
            sampler(3, 1f, 0, 1f, 1f, 7).sampleToken(t);
            assertEquals(Set.of(0), positiveIds(t));
        }
    }

    @Nested
    class FullStack {

        @Test
        void samplesOnlyFromTheFilteredSet() {
            // top-k 2 keeps ids {1,3}; the sampled token must always come from that set
            Sampler stack = sampler(5, 0.8f, 2, 0.95f, 0.05f, 7);
            for (int trial = 0; trial < 100; trial++) {
                var t = logits(1f, 5f, 3f, 4f, 2f);
                int token = stack.sampleToken(t);
                assertTrue(token == 1 || token == 3, "sampled " + token);
            }
        }

        @Test
        void nucleusRunsBeforeMinP() {
            // Weights {0.5, 0.3, 0.2}: top-p 0.6 keeps the first two, then min-p 0.5 keeps both.
            // Reversing the filters renormalizes {0.5, 0.3}, letting top-p keep only the first.
            Sampler stack = sampler(4, 1f, 3, 0.6f, 0.5f, 7);
            var seen = new HashSet<Integer>();
            for (int i = 0; i < 300; i++) {
                seen.add(
                        stack.sampleToken(
                                logits(
                                        (float) Math.log(0.5),
                                        (float) Math.log(0.3),
                                        (float) Math.log(0.2),
                                        -100f)));
            }
            assertEquals(Set.of(0, 1), seen);
        }

        @Test
        void filtersRunBeforeTemperature() {
            // The unscaled top probability is 0.5, so top-p 0.45 retains only token 0. Applying
            // temperature first would flatten the distribution and incorrectly retain token 1.
            Sampler stack = sampler(4, 5f, 3, 0.45f, 0f, 7);
            for (int i = 0; i < 100; i++) {
                assertEquals(
                        0,
                        stack.sampleToken(
                                logits(
                                        (float) Math.log(0.5),
                                        (float) Math.log(0.3),
                                        (float) Math.log(0.2),
                                        -100f)));
            }
        }

        @Test
        void sameSeedSameSequence() {
            Sampler a = sampler(5, 1.2f, 0, 0.99f, 0f, 1234);
            Sampler b = sampler(5, 1.2f, 0, 0.99f, 0f, 1234);
            for (int i = 0; i < 50; i++) {
                assertEquals(
                        a.sampleToken(logits(1f, 2f, 3f, 2f, 1f)),
                        b.sampleToken(logits(1f, 2f, 3f, 2f, 1f)));
            }
        }

        @Test
        void differentSeedsDiverge() {
            Sampler a = sampler(4, 5f, 0, 1f, 0f, 1);
            Sampler b = sampler(4, 5f, 0, 1f, 0f, 2);
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
            Sampler s = sampler(4, 1f, 0, 1f, 0f, 99);
            int peak = 0;
            for (int i = 0; i < 200; i++) {
                if (s.sampleToken(logits(6f, 0f, 0f, 0f)) == 0) peak++;
            }
            assertTrue(peak > 190, "peak sampled " + peak + "/200"); // p(peak) ~ 0.993
        }

        @Test
        void categoricalFrequenciesTrackProbabilities() {
            // ln(2) logit gap = 2:1 odds between ids 0 and 1
            Sampler s = sampler(2, 1f, 0, 1f, 0f, 5);
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
