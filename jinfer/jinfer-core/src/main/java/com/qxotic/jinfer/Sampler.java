package com.qxotic.jinfer;

import java.util.Comparator;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Token sampling over a logits tensor, with composable building blocks: {@link #ARGMAX},
 * {@link CategoricalSampler}, {@link ToppSampler}, the {@link #withTemperature} softmax wrapper
 * and the {@link #banning} token filter. {@link #select} assembles the standard stack from
 * (temperature, top-p, seed); model-aware policy (think-token bans) lives in
 * {@link Engine#configuredSampler}.
 */
@FunctionalInterface
interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;

    /** The standard sampling stack: greedy at temperature 0, otherwise temperature-scaled
     *  softmax feeding categorical sampling (top-p nucleus sampling when 0 &lt; topp &lt; 1). */
    static Sampler select(int vocabularySize, float temperature, float topp, long rngSeed) {
        if (temperature == 0.0f) {
            return ARGMAX;
        }
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
        Sampler innerSampler;
        if (topp <= 0 || topp >= 1) {
            innerSampler = new CategoricalSampler(rng);
        } else {
            innerSampler = new ToppSampler(vocabularySize, topp, rng);
        }
        return withTemperature(innerSampler, temperature);
    }

    /** Temperature-scales the logits and converts them to probabilities (in place) before
     *  delegating; the inner sampler sees a probability distribution. */
    static Sampler withTemperature(Sampler inner, float temperature) {
        return logits -> {
            int logitsSize = Math.toIntExact(logits.size());
            logits.divideInPlace(0, logitsSize, temperature);
            logits.softmaxInPlace(0, logitsSize);
            return inner.sampleToken(logits);
        };
    }

    /** Makes the given tokens unsamplable by forcing their logits to -inf before delegating. */
    static Sampler banning(Sampler inner, Set<Integer> bannedTokens) {
        if (bannedTokens.isEmpty()) {
            return inner;
        }
        return logits -> {
            for (int token : bannedTokens) logits.setFloat(token, Float.NEGATIVE_INFINITY);
            return inner.sampleToken(logits);
        };
    }

    /** Grammar-constrained sampling: masks logits with the current grammar state before
     *  delegating, then advances the grammar with the chosen token. When no valid token
     *  remains, forces {@code eosToken} so generation terminates cleanly instead of
     *  feeding a garbage token into the forward pass. */
    static Sampler withGrammar(Sampler inner, Grammar.Cursor cursor, int eosToken) {
        if (cursor == null || !RuntimeFlags.GRAMMAR) return inner;
        return logits -> {
            if (!cursor.maskLogits(logits)) {
                cursor.advanceWith(eosToken);
                return eosToken;
            }
            int token = inner.sampleToken(logits);
            cursor.advanceWith(token);
            return token;
        };
    }
}

record CategoricalSampler(RandomGenerator rng) implements Sampler {

    @Override
    public int sampleToken(FloatTensor logits) {
        float random0to1 = rng.nextFloat(1f);
        float cdf = 0.0f;
        for (int i = 0; i < logits.size(); i++) {
            cdf += logits.getFloat(i);
            if (random0to1 < cdf) {
                return i;
            }
        }
        return Math.toIntExact(logits.size()) - 1;
    }
}

final class ToppSampler implements Sampler {

    final int[] indices;
    final float topp;
    final RandomGenerator rng;

    public ToppSampler(int maxNumberOfElements, float topp, RandomGenerator rng) {
        this.indices = new int[maxNumberOfElements];
        this.topp = topp;
        this.rng = rng;
    }

    static void swap(int[] array, int from, int to) {
        int tmp = array[from];
        array[from] = array[to];
        array[to] = tmp;
    }

    static void siftDown(int[] array, int from, int n, Comparator<Integer> comparator) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = 2 * prev + 2;
            if (r < n && comparator.compare(array[r], array[next]) < 0) {
                next = r;
            }
            if (comparator.compare(array[next], array[prev]) < 0) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    @Override
    public int sampleToken(FloatTensor logits) {
        Comparator<Integer> comparator = Comparator.comparingDouble((Integer i) -> logits.getFloat(i)).reversed();

        int n = Math.toIntExact(logits.size());
        int head = 0;
        int tail = n - 1;
        float cutoff = (1.0f - topp) / (n - 1);
        for (int i = 0; i < indices.length; i++) {
            if (logits.getFloat(i) >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int n0 = head;
        for (int i = n0 / 2 - 1; i >= 0; --i) {
            siftDown(indices, i, n0, comparator);
        }

        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = n0 - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += logits.getFloat(indices[i]);
            if (cumulativeProb > topp) {
                lastIndex = i;
                break;
            }
            siftDown(indices, 0, i, comparator);
        }

        float r = rng.nextFloat(1f) * cumulativeProb;
        float cdf = 0.0f;
        for (int i = n0 - 1; i >= lastIndex; i--) {
            cdf += logits.getFloat(indices[i]);
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex];
    }
}
