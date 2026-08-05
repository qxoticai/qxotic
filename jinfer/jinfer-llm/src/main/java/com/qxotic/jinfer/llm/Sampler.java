package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.*;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/**
 * Token sampling over a logits tensor, with composable building blocks: {@link #ARGMAX}, {@link
 * CategoricalSampler}, the filter wrappers and the {@link #withTemperature} softmax wrapper. {@link
 * #select} assembles the standard stack in llama.cpp's chain order - top-k, top-p, min-p filter the
 * logits, then temperature scales what survives and a categorical draw picks the token - so a given
 * (temperature, top-k, top-p, min-p) samples the same token population as llama.cpp. Model-aware
 * policy (think-token bans, grammars) is the caller's to layer on top.
 */
@FunctionalInterface
public interface Sampler {
    int sampleToken(FloatTensor logits);

    Sampler ARGMAX = FloatTensor::argmax;

    /**
     * The standard stack. Temperature 0 is greedy argmax outright - every filter keeps the
     * highest-logit token, so filtering before argmax changes nothing. Disabled values follow
     * llama.cpp: {@code topK <= 0} or {@code >= vocabularySize} means no top-k, {@code topP >= 1}
     * no top-p, {@code minP <= 0} no min-p.
     */
    static Sampler select(
            int vocabularySize, float temperature, int topK, float topP, float minP, long rngSeed) {
        if (temperature == 0.0f) {
            return ARGMAX;
        }
        RandomGenerator rng = RandomGeneratorFactory.getDefault().create(rngSeed);
        Sampler stack = withTemperature(new CategoricalSampler(rng), temperature);
        // built inside-out: the OUTERMOST wrapper runs first, so this is llama.cpp's
        // top-k -> top-p -> min-p -> temperature order
        if (minP > 0 && minP < 1) {
            stack = withMinP(stack, minP);
        }
        if (topP > 0 && topP < 1) {
            stack = new NucleusFilter(vocabularySize, topP, stack);
        }
        if (topK > 0 && topK < vocabularySize) {
            stack = withTopK(stack, topK);
        }
        return stack;
    }

    /**
     * Keeps the {@code k} highest logits and masks the rest to -inf before delegating. Ties at the
     * k-th value all survive (llama.cpp cuts ties arbitrarily by sort order; keeping them is the
     * deterministic reading of the same contract).
     */
    static Sampler withTopK(Sampler inner, int k) {
        float[] heap = new float[k]; // min-heap of the k largest logits seen so far
        return logits -> {
            int n = Math.toIntExact(logits.size());
            java.util.Arrays.fill(heap, Float.NEGATIVE_INFINITY);
            for (int i = 0; i < n; i++) {
                float v = logits.getFloat(i);
                if (v > heap[0]) {
                    heap[0] = v;
                    int prev = 0, next;
                    while ((next = 2 * prev + 1) < k) {
                        int r = next + 1;
                        if (r < k && heap[r] < heap[next]) next = r;
                        if (heap[next] < heap[prev]) {
                            float t = heap[prev];
                            heap[prev] = heap[next];
                            heap[next] = t;
                            prev = next;
                        } else {
                            break;
                        }
                    }
                }
            }
            float threshold = heap[0];
            for (int i = 0; i < n; i++) {
                if (logits.getFloat(i) < threshold) {
                    logits.setFloat(i, Float.NEGATIVE_INFINITY);
                }
            }
            return inner.sampleToken(logits);
        };
    }

    /**
     * Masks tokens whose probability falls below {@code minP} times the top token's probability
     * (computed on logits: {@code logit < maxLogit + ln(minP)}) before delegating.
     */
    static Sampler withMinP(Sampler inner, float minP) {
        float logMinP = (float) Math.log(minP);
        return logits -> {
            int n = Math.toIntExact(logits.size());
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < n; i++) {
                max = Math.max(max, logits.getFloat(i));
            }
            float threshold = max + logMinP;
            for (int i = 0; i < n; i++) {
                if (logits.getFloat(i) < threshold) {
                    logits.setFloat(i, Float.NEGATIVE_INFINITY);
                }
            }
            return inner.sampleToken(logits);
        };
    }

    /**
     * Temperature-scales the logits and converts them to probabilities (in place) before
     * delegating; the inner sampler sees a probability distribution.
     */
    private static Sampler withTemperature(Sampler inner, float temperature) {
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
        int[] banned = bannedTokens.stream().mapToInt(Integer::intValue).toArray();
        return logits -> {
            for (int token : banned) logits.setFloat(token, Float.NEGATIVE_INFINITY);
            return inner.sampleToken(logits);
        };
    }

    /**
     * Prefix-pin with a forced epilogue: once the pin exhausts, {@code epilogue} ids are emitted
     * verbatim (bypassing the logits) before sampling releases. For families whose call header
     * continues with SCAFFOLD after the pinned name - Harmony's {@code " <|constrain|>json
     * <|message|>"} - improvising it from an off-policy state derails generation; forcing it keeps
     * the model on-distribution until the free payload.
     */
    static Sampler withPrefixGrammar(
            Sampler inner, Grammar.Cursor cursor, int eosToken, int[] epilogue) {
        if (cursor == null || !RuntimeFlags.GRAMMAR) return inner;
        boolean[] released = {false};
        int[] forced = {0};
        return logits -> {
            if (released[0]) return inner.sampleToken(logits);
            if (cursor.exhausted()) {
                if (forced[0] < epilogue.length) return epilogue[forced[0]++];
                released[0] = true;
                return inner.sampleToken(logits);
            }
            if (!cursor.maskLogits(logits)) {
                cursor.advanceWith(eosToken); // no vocab token fits the pin: end cleanly
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

final class NucleusFilter implements Sampler {

    // the one buffer, sized once at construction: candidate token ids, heap-ordered in place
    private final int[] candidates;
    private final float topP;
    private final Sampler inner;

    NucleusFilter(int vocabularySize, float topP, Sampler inner) {
        this.candidates = new int[vocabularySize];
        this.topP = topP;
        this.inner = inner;
    }

    private void swap(int from, int to) {
        int tmp = candidates[from];
        candidates[from] = candidates[to];
        candidates[to] = tmp;
    }

    /** Max-heap sift on candidate ids ordered by their logits - primitives only, no boxing. */
    private void siftDown(int from, int n, FloatTensor logits) {
        int prev = from, next;
        while ((next = 2 * prev + 1) < n) {
            int r = next + 1;
            if (r < n && logits.getFloat(candidates[r]) > logits.getFloat(candidates[next])) {
                next = r;
            }
            if (logits.getFloat(candidates[next]) > logits.getFloat(candidates[prev])) {
                swap(prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    /**
     * Masks everything outside the smallest set of tokens whose probabilities sum past {@code topP}
     * (the last token crossing the line is kept, llama.cpp's cut), then delegates. Works on raw
     * logits - probabilities are derived on the fly so the tensor still holds logits for the
     * temperature stage downstream. Heap extraction reorders {@code candidates} in place (the
     * nucleus ends up in its tail), so the only buffer is the candidate ids, sized once at
     * construction; the per-token path allocates nothing.
     */
    @Override
    public int sampleToken(FloatTensor logits) {
        int n = Math.toIntExact(logits.size());
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < n; i++) {
            max = Math.max(max, logits.getFloat(i));
        }
        double denom = 0;
        for (int i = 0; i < n; i++) {
            denom = denom + Math.exp(logits.getFloat(i) - max);
        }
        // tokens below probability (1-topP)/(n-1) cannot be part of any nucleus of mass topP;
        // the same cut in the logit domain (one log instead of n exps) masks them outright, and
        // they never enter the candidate set
        float cutoff = (float) (max + Math.log((1.0 - topP) / (n - 1) * denom));
        int head = 0;
        for (int i = 0; i < n; i++) {
            if (logits.getFloat(i) >= cutoff) {
                candidates[head++] = i;
            } else {
                logits.setFloat(i, Float.NEGATIVE_INFINITY);
            }
        }
        for (int i = head / 2 - 1; i >= 0; --i) {
            siftDown(i, head, logits);
        }
        // extract-max parks each nucleus token at the shrinking tail: after the loop the nucleus
        // occupies candidates[remaining..head) and the also-rans candidates[0..remaining)
        double cumulative = 0;
        int remaining = head;
        while (remaining > 0 && cumulative < topP) {
            int id = candidates[0];
            cumulative += Math.exp(logits.getFloat(id) - max) / denom;
            swap(0, --remaining);
            siftDown(0, remaining, logits);
        }
        for (int i = 0; i < remaining; i++) {
            logits.setFloat(candidates[i], Float.NEGATIVE_INFINITY);
        }
        return inner.sampleToken(logits);
    }
}
