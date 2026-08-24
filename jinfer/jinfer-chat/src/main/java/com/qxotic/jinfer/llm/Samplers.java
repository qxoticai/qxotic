package com.qxotic.jinfer.llm;

import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.concurrent.ThreadLocalRandom;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/** Built-in sampling implementations. */
final class Samplers {
    private Samplers() {}

    static Sampler create(Sampling sampling, int vocabularySize) {
        if (vocabularySize <= 0)
            throw new IllegalArgumentException("vocabularySize " + vocabularySize);
        if (sampling.temperature() == 0) return Sampler.ARGMAX;

        long seed =
                sampling.seed() != null ? sampling.seed() : ThreadLocalRandom.current().nextLong();
        if (sampling.topK() > 0 && sampling.topK() < vocabularySize) {
            return compact(sampling, vocabularySize, seed);
        }
        return dense(sampling, vocabularySize, seed);
    }

    private static Sampler compact(Sampling sampling, int vocabularySize, long seed) {
        int[] tokenIds = new int[sampling.topK()];
        float[] logits = new float[sampling.topK()];
        float topP = sampling.topP();
        boolean minPEnabled = sampling.minP() > 0;
        float logMinP = minPEnabled ? (float) Math.log(sampling.minP()) : 0;
        float inverseTemperature = 1 / sampling.temperature();
        RandomGenerator random = RandomGeneratorFactory.getDefault().create(seed);

        return input -> {
            MemoryView<MemorySegment> values = checked(input, vocabularySize);
            selectTopK(values, tokenIds, logits);
            int size = tokenIds.length;
            if (topP < 1) size = retainTopP(logits, size, topP);
            if (minPEnabled) size = retainMinP(logits, size, logMinP);
            return categorical(tokenIds, logits, size, inverseTemperature, random);
        };
    }

    private static Sampler dense(Sampling sampling, int vocabularySize, long seed) {
        float topP = sampling.topP();
        float minP = sampling.minP();
        float temperature = sampling.temperature();
        int[] candidates = topP < 1 ? new int[vocabularySize] : null;
        float logMinP = minP > 0 ? (float) Math.log(minP) : 0;
        RandomGenerator random = RandomGeneratorFactory.getDefault().create(seed);

        return input -> {
            MemoryView<MemorySegment> logits = checked(input, vocabularySize);
            if (candidates != null) maskTopP(logits, candidates, topP);
            if (minP > 0) maskBelow(logits, max(logits) + logMinP);
            Ops.divideInPlace(logits, 0, vocabularySize, temperature);
            Ops.softmaxInPlace(logits, 0, vocabularySize);
            return categorical(logits, random);
        };
    }

    /** Selects exactly {@code tokenIds.length} candidates, ordered by logit then token id. */
    static void selectTopK(
            MemoryView<MemorySegment> values, int[] tokenIds, float[] candidateLogits) {
        int size = Math.toIntExact(values.shape().size());
        int k = tokenIds.length;

        MemorySegment segment = values.memory().base();
        long base = values.byteOffset();
        for (int token = 0; token < k; token++) {
            tokenIds[token] = token;
            candidateLogits[token] =
                    segment.get(ValueLayout.JAVA_FLOAT, base + (long) token * Float.BYTES);
        }
        for (int at = (k >>> 1) - 1; at >= 0; at--) {
            siftDown(tokenIds, candidateLogits, at, k, tokenIds[at], candidateLogits[at]);
        }

        for (int token = k; token < size; token++) {
            float logit = segment.get(ValueLayout.JAVA_FLOAT, base + (long) token * Float.BYTES);
            if (!better(logit, token, candidateLogits[0], tokenIds[0])) continue;
            siftDown(tokenIds, candidateLogits, 0, k, token, logit);
        }

        sortDescending(tokenIds, candidateLogits);
    }

    private static boolean better(float logit, int token, float otherLogit, int otherToken) {
        return logit > otherLogit || (logit == otherLogit && token < otherToken);
    }

    private static boolean worse(float logit, int token, float otherLogit, int otherToken) {
        return logit < otherLogit || (logit == otherLogit && token > otherToken);
    }

    /** Restores the min-heap whose root is the worst retained candidate. */
    private static void siftDown(
            int[] tokenIds, float[] logits, int at, int size, int token, float logit) {
        int firstLeaf = size >>> 1;
        while (at < firstLeaf) {
            int child = (at << 1) + 1;
            int right = child + 1;
            if (right < size
                    && worse(logits[right], tokenIds[right], logits[child], tokenIds[child]))
                child = right;
            if (!worse(logits[child], tokenIds[child], logit, token)) break;
            logits[at] = logits[child];
            tokenIds[at] = tokenIds[child];
            at = child;
        }
        logits[at] = logit;
        tokenIds[at] = token;
    }

    private static void sortDescending(int[] tokenIds, float[] logits) {
        for (int end = logits.length - 1; end > 0; end--) {
            int worstToken = tokenIds[0];
            float worstLogit = logits[0];
            int replacementToken = tokenIds[end];
            float replacementLogit = logits[end];
            tokenIds[end] = worstToken;
            logits[end] = worstLogit;
            siftDown(tokenIds, logits, 0, end, replacementToken, replacementLogit);
        }
    }

    static int retainTopP(float[] sortedLogits, int size, float topP) {
        float max = sortedLogits[0];
        double denominator = 0;
        for (int i = 0; i < size; i++) denominator += Math.exp(sortedLogits[i] - max);

        double target = topP * denominator;
        double cumulative = 0;
        int kept = 0;
        do {
            cumulative += Math.exp(sortedLogits[kept] - max);
        } while (++kept < size && cumulative < target);
        return kept;
    }

    static int retainMinP(float[] sortedLogits, int size, float logMinP) {
        float threshold = sortedLogits[0] + logMinP;
        while (size > 1 && sortedLogits[size - 1] < threshold) size--;
        return size;
    }

    private static int categorical(
            int[] tokenIds,
            float[] logits,
            int size,
            float inverseTemperature,
            RandomGenerator random) {
        float max = logits[0];
        double sum = 0;
        for (int i = 0; i < size; i++) {
            float weight = (float) Math.exp((logits[i] - max) * inverseTemperature);
            logits[i] = weight;
            sum += weight;
        }

        double draw = random.nextDouble(sum);
        for (int i = 0; i < size; i++) {
            draw -= logits[i];
            if (draw < 0) return tokenIds[i];
        }
        return tokenIds[size - 1];
    }

    private static MemoryView<MemorySegment> checked(MemoryView<?> logits, int vocabularySize) {
        Views.requireF32(logits, "logits");
        Views.requireContiguous(logits, "logits");
        MemoryView<MemorySegment> values = Views.castToSegmentBacked(logits, "logits");
        Views.checkAlive(values, "logits");
        if (values.shape().size() != vocabularySize)
            throw new IllegalArgumentException(
                    "logits size "
                            + values.shape().size()
                            + " != vocabularySize "
                            + vocabularySize);
        return values;
    }

    private static float get(MemoryView<MemorySegment> values, int index) {
        return values.memory()
                .base()
                .get(ValueLayout.JAVA_FLOAT, values.byteOffset() + (long) index * Float.BYTES);
    }

    private static void set(MemoryView<MemorySegment> values, int index, float value) {
        values.memory()
                .base()
                .set(
                        ValueLayout.JAVA_FLOAT,
                        values.byteOffset() + (long) index * Float.BYTES,
                        value);
    }

    private static float max(MemoryView<MemorySegment> values) {
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0, n = Math.toIntExact(values.shape().size()); i < n; i++)
            max = Math.max(max, get(values, i));
        return max;
    }

    private static void maskBelow(MemoryView<MemorySegment> values, float threshold) {
        for (int i = 0, n = Math.toIntExact(values.shape().size()); i < n; i++)
            if (get(values, i) < threshold) set(values, i, Float.NEGATIVE_INFINITY);
    }

    private static int categorical(
            MemoryView<MemorySegment> probabilities, RandomGenerator random) {
        float draw = random.nextFloat(1f);
        float cumulative = 0;
        int size = Math.toIntExact(probabilities.shape().size());
        for (int i = 0; i < size; i++) {
            cumulative += get(probabilities, i);
            if (draw < cumulative) return i;
        }
        for (int i = size - 1; i >= 0; i--) if (get(probabilities, i) > 0) return i;
        return size - 1;
    }

    private static void maskTopP(MemoryView<MemorySegment> logits, int[] candidates, float topP) {
        int size = candidates.length;
        float max = max(logits);
        double denominator = 0;
        for (int i = 0; i < size; i++) denominator += Math.exp(get(logits, i) - max);
        float cutoff =
                size == 1
                        ? max
                        : Math.min(
                                max,
                                (float) (max + Math.log((1.0 - topP) / (size - 1) * denominator)));
        int count = 0;
        for (int i = 0; i < size; i++) {
            if (get(logits, i) >= cutoff) candidates[count++] = i;
            else set(logits, i, Float.NEGATIVE_INFINITY);
        }
        for (int i = count / 2 - 1; i >= 0; i--) siftDown(logits, candidates, i, count);
        double cumulative = 0;
        int remaining = count;
        while (remaining > 0 && cumulative < topP) {
            cumulative += Math.exp(get(logits, candidates[0]) - max) / denominator;
            swap(candidates, 0, --remaining);
            siftDown(logits, candidates, 0, remaining);
        }
        for (int i = 0; i < remaining; i++) set(logits, candidates[i], Float.NEGATIVE_INFINITY);
    }

    private static void siftDown(
            MemoryView<MemorySegment> logits, int[] candidates, int from, int size) {
        for (int at = from; ; ) {
            int child = at * 2 + 1;
            if (child >= size) return;
            if (child + 1 < size
                    && get(logits, candidates[child + 1]) > get(logits, candidates[child])) child++;
            if (get(logits, candidates[child]) <= get(logits, candidates[at])) return;
            swap(candidates, at, child);
            at = child;
        }
    }

    private static void swap(int[] values, int a, int b) {
        int value = values[a];
        values[a] = values[b];
        values[b] = value;
    }
}
