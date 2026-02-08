package ai.qxotic.jota.examples.llama;

import ai.qxotic.jota.DataType;
import ai.qxotic.jota.Environment;
import ai.qxotic.jota.memory.MemoryAccess;
import ai.qxotic.jota.memory.MemoryView;
import java.util.Random;

final class Sampler {
    private final int vocab;
    private final float temperature;
    private final float topP;
    private final Random rng;
    private final float[] probs;
    private final int[] indices;

    Sampler(int vocab, float temperature, float topP, long seed) {
        this.vocab = vocab;
        this.temperature = temperature;
        this.topP = topP;
        this.rng = new Random(seed);
        this.probs = new float[vocab];
        this.indices = new int[vocab];
        for (int i = 0; i < vocab; i++) {
            indices[i] = i;
        }
    }

    int sample(float[] logits) {
        if (temperature <= 0f) {
            return argmax(logits);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            if (logits[i] > max) {
                max = logits[i];
            }
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            double p = Math.exp((logits[i] - max) / temperature);
            probs[i] = (float) p;
            sum += p;
        }
        for (int i = 0; i < vocab; i++) {
            probs[i] /= (float) sum;
        }
        if (topP > 0f && topP < 1f) {
            return sampleTopP(probs, topP);
        }
        return sampleCategorical(probs);
    }

    int sample(MemoryView<?> logitsView) {
        if (logitsView.dataType() != DataType.FP32 || !logitsView.layout().isSuffixContiguous(0)) {
            throw new IllegalArgumentException("Sampler expects FP32 contiguous logits view");
        }
        @SuppressWarnings({"rawtypes", "unchecked"})
        MemoryAccess access =
                Environment.current().runtimeFor(logitsView.memory().device()).memoryDomain().directAccess();
        if (access == null) {
            throw new IllegalStateException("No direct memory access for logits view");
        }
        long base = logitsView.byteOffset();
        if (temperature <= 0f) {
            return argmax(logitsView, access, base);
        }
        float max = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < vocab; i++) {
            float v = access.readFloat(logitsView.memory(), base + (long) i * Float.BYTES);
            if (v > max) {
                max = v;
            }
        }
        double sum = 0;
        for (int i = 0; i < vocab; i++) {
            float logit = access.readFloat(logitsView.memory(), base + (long) i * Float.BYTES);
            double p = Math.exp((logit - max) / temperature);
            probs[i] = (float) p;
            sum += p;
        }
        for (int i = 0; i < vocab; i++) {
            probs[i] /= (float) sum;
        }
        if (topP > 0f && topP < 1f) {
            return sampleTopP(probs, topP);
        }
        return sampleCategorical(probs);
    }

    private int sampleCategorical(float[] probs) {
        float r = rng.nextFloat();
        float c = 0f;
        for (int i = 0; i < probs.length; i++) {
            c += probs[i];
            if (r <= c) {
                return i;
            }
        }
        return probs.length - 1;
    }

    private int sampleTopP(float[] probs, float p) {
        int n = probs.length;
        if (n == 0) {
            return 0;
        }
        if (n == 1) {
            return 0;
        }

        int head = 0;
        int tail = n - 1;
        float cutoff = (1.0f - p) / (n - 1);
        for (int i = 0; i < n; i++) {
            if (probs[i] >= cutoff) {
                indices[head++] = i;
            } else {
                indices[tail--] = i;
            }
        }

        int candidateCount = head;
        if (candidateCount <= 0) {
            return argmax(probs);
        }

        for (int i = candidateCount / 2 - 1; i >= 0; --i) {
            siftDownByProb(indices, i, candidateCount, probs);
        }

        float cumulativeProb = 0.0f;
        int lastIndex = 0;
        for (int i = candidateCount - 1; i >= 0; i--) {
            swap(indices, 0, i);
            cumulativeProb += probs[indices[i]];
            if (cumulativeProb > p) {
                lastIndex = i;
                break;
            }
            siftDownByProb(indices, 0, i, probs);
        }

        float r = rng.nextFloat() * cumulativeProb;
        float cdf = 0f;
        for (int i = candidateCount - 1; i >= lastIndex; i--) {
            cdf += probs[indices[i]];
            if (r < cdf) {
                return indices[i];
            }
        }

        return indices[lastIndex];
    }

    private static void swap(int[] array, int from, int to) {
        int t = array[from];
        array[from] = array[to];
        array[to] = t;
    }

    private static void siftDownByProb(int[] array, int from, int n, float[] probs) {
        int prev = from;
        while (true) {
            int left = 2 * prev + 1;
            if (left >= n) {
                break;
            }
            int right = left + 1;
            int next = left;
            if (right < n && probs[array[right]] > probs[array[left]]) {
                next = right;
            }
            if (probs[array[next]] > probs[array[prev]]) {
                swap(array, prev, next);
                prev = next;
            } else {
                break;
            }
        }
    }

    private static int argmax(float[] x) {
        int best = 0;
        for (int i = 1; i < x.length; i++) {
            if (x[i] > x[best]) {
                best = i;
            }
        }
        return best;
    }

    @SuppressWarnings("rawtypes")
    private int argmax(MemoryView<?> logitsView, MemoryAccess access, long base) {
        int best = 0;
        float bestValue = access.readFloat(logitsView.memory(), base);
        for (int i = 1; i < vocab; i++) {
            float v = access.readFloat(logitsView.memory(), base + (long) i * Float.BYTES);
            if (v > bestValue) {
                bestValue = v;
                best = i;
            }
        }
        return best;
    }
}
