package com.qxotic.jinfer.x.llm;

import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;
import java.util.Objects;
import java.util.Set;
import java.util.random.RandomGenerator;
import java.util.random.RandomGeneratorFactory;

/** Composable token sampling over a dense FP32 logits view. */
@FunctionalInterface
public interface Sampler {
    int sampleToken(MemoryView<?> logits);

    Sampler ARGMAX =
            logits -> {
                MemoryView<MemorySegment> values = checked(logits);
                return Ops.argmax(values, 0, size(values));
            };

    static Sampler select(
            int vocabularySize, float temperature, int topK, float topP, float minP, long seed) {
        if (vocabularySize <= 0)
            throw new IllegalArgumentException("vocabularySize " + vocabularySize);
        if (temperature == 0) return ARGMAX;
        Sampler sampler =
                withTemperature(
                        categorical(RandomGeneratorFactory.getDefault().create(seed)), temperature);
        if (minP > 0 && minP <= 1) sampler = withMinP(sampler, minP);
        if (topP > 0 && topP < 1) sampler = nucleus(vocabularySize, topP, sampler);
        if (topK > 0 && topK < vocabularySize) sampler = withTopK(sampler, topK);
        return sampler;
    }

    static Sampler withTopK(Sampler inner, int k) {
        Objects.requireNonNull(inner, "inner");
        if (k <= 0) throw new IllegalArgumentException("k " + k);
        float[] heap = new float[k];
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            Arrays.fill(heap, Float.NEGATIVE_INFINITY);
            int n = size(values);
            for (int i = 0; i < n; i++) {
                float value = get(values, i);
                if (value > heap[0]) replaceMin(heap, value);
            }
            maskBelow(values, heap[0]);
            return inner.sampleToken(values);
        };
    }

    static Sampler withMinP(Sampler inner, float minP) {
        Objects.requireNonNull(inner, "inner");
        if (!(minP > 0 && minP <= 1)) throw new IllegalArgumentException("minP " + minP);
        float logMinP = (float) Math.log(minP);
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            maskBelow(values, max(values) + logMinP);
            return inner.sampleToken(values);
        };
    }

    private static Sampler withTemperature(Sampler inner, float temperature) {
        if (!(temperature > 0)) throw new IllegalArgumentException("temperature " + temperature);
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            int n = size(values);
            Ops.divideInPlace(values, 0, n, temperature);
            Ops.softmaxInPlace(values, 0, n);
            return inner.sampleToken(values);
        };
    }

    static Sampler banning(Sampler inner, Set<Integer> bannedTokens) {
        Objects.requireNonNull(inner, "inner");
        Objects.requireNonNull(bannedTokens, "bannedTokens");
        if (bannedTokens.isEmpty()) return inner;
        int[] banned = bannedTokens.stream().mapToInt(Integer::intValue).toArray();
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            int n = size(values);
            for (int token : banned) {
                if (token < 0 || token >= n)
                    throw new IllegalArgumentException("banned token " + token + " outside logits");
                set(values, token, Float.NEGATIVE_INFINITY);
            }
            return inner.sampleToken(values);
        };
    }

    private static MemoryView<MemorySegment> checked(MemoryView<?> logits) {
        Views.requireF32(logits, "logits");
        Views.requireContiguous(logits, "logits");
        MemoryView<MemorySegment> values = Views.castToSegmentBacked(logits, "logits");
        Views.checkAlive(values, "logits");
        if (values.shape().size() == 0) throw new IllegalArgumentException("empty logits");
        return values;
    }

    private static int size(MemoryView<?> values) {
        return Math.toIntExact(values.shape().size());
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
        for (int i = 0, n = size(values); i < n; i++) max = Math.max(max, get(values, i));
        return max;
    }

    private static void maskBelow(MemoryView<MemorySegment> values, float threshold) {
        for (int i = 0, n = size(values); i < n; i++)
            if (get(values, i) < threshold) set(values, i, Float.NEGATIVE_INFINITY);
    }

    private static void replaceMin(float[] heap, float value) {
        heap[0] = value;
        for (int at = 0; ; ) {
            int child = at * 2 + 1;
            if (child >= heap.length) return;
            if (child + 1 < heap.length && heap[child + 1] < heap[child]) child++;
            if (heap[child] >= heap[at]) return;
            float swap = heap[at];
            heap[at] = heap[child];
            heap[child] = swap;
            at = child;
        }
    }

    private static Sampler categorical(RandomGenerator random) {
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            float draw = random.nextFloat(1f);
            float cumulative = 0;
            int n = size(values);
            for (int i = 0; i < n; i++) {
                cumulative += get(values, i);
                if (draw < cumulative) return i;
            }
            for (int i = n - 1; i >= 0; i--) if (get(values, i) > 0) return i;
            return n - 1;
        };
    }

    private static Sampler nucleus(int vocabularySize, float topP, Sampler inner) {
        int[] candidates = new int[vocabularySize];
        return logits -> {
            MemoryView<MemorySegment> values = checked(logits);
            int n = size(values);
            if (n != candidates.length)
                throw new IllegalArgumentException(
                        "logits size " + n + " != vocabularySize " + candidates.length);
            float max = max(values);
            double denominator = 0;
            for (int i = 0; i < n; i++) denominator += Math.exp(get(values, i) - max);
            float cutoff =
                    n == 1 ? max : (float) (max + Math.log((1.0 - topP) / (n - 1) * denominator));
            int count = 0;
            for (int i = 0; i < n; i++) {
                if (get(values, i) >= cutoff) candidates[count++] = i;
                else set(values, i, Float.NEGATIVE_INFINITY);
            }
            for (int i = count / 2 - 1; i >= 0; i--) siftDown(candidates, i, count, values);
            double cumulative = 0;
            int remaining = count;
            while (remaining > 0 && cumulative < topP) {
                cumulative += Math.exp(get(values, candidates[0]) - max) / denominator;
                swap(candidates, 0, --remaining);
                siftDown(candidates, 0, remaining, values);
            }
            for (int i = 0; i < remaining; i++) set(values, candidates[i], Float.NEGATIVE_INFINITY);
            return inner.sampleToken(values);
        };
    }

    private static void siftDown(
            int[] candidates, int from, int size, MemoryView<MemorySegment> logits) {
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
