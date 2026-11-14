package com.qxotic.jinfer.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Views;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import org.junit.jupiter.api.Test;

/** Behavioral contracts for cached-prefix prefill, independent of the retired implementation. */
class FlashAttentionContractTest {

    private static final int HEADS = 4;
    private static final int KV_MUL = 2;
    private static final int HEAD_SIZE = 64;
    private static final int KV_DIM = (HEADS / KV_MUL) * HEAD_SIZE;
    private static final int QUERY_DIM = HEADS * HEAD_SIZE;

    private final MemoryArena<MemorySegment> memory = MemoryAllocators.ofArena(Arena.ofAuto());

    @Test
    void chunkedPrefillMatchesSingleShotAtDepth() {
        int prefix = 96, tail = 61, length = prefix + tail;
        MemoryView<MemorySegment> q = randomF32(length * QUERY_DIM, 1);
        MemoryView<MemorySegment> k = randomF32(length * KV_DIM, 2);
        MemoryView<MemorySegment> v = randomF32(length * KV_DIM, 3);
        MemoryView<MemorySegment> empty = Views.allocateF32(memory, 1);

        MemoryView<MemorySegment> single = Views.allocateF32(memory, length * QUERY_DIM);
        prefill(q, single, empty, empty, k, v, 0, length);

        MemoryView<MemorySegment> qTail = slice(q, (long) prefix * QUERY_DIM, tail * QUERY_DIM);
        MemoryView<MemorySegment> kTail = slice(k, (long) prefix * KV_DIM, tail * KV_DIM);
        MemoryView<MemorySegment> vTail = slice(v, (long) prefix * KV_DIM, tail * KV_DIM);
        MemoryView<MemorySegment> chunked = Views.allocateF32(memory, tail * QUERY_DIM);
        prefill(qTail, chunked, k, v, kTail, vTail, prefix, tail);

        assertClose(slice(single, (long) prefix * QUERY_DIM, tail * QUERY_DIM), chunked);
    }

    @Test
    void f16CacheMatchesItsWidenedF32Values() {
        int prefix = 96, tail = 61, elements = prefix * KV_DIM;
        MemoryView<MemorySegment> q = randomF32(tail * QUERY_DIM, 21);
        MemoryView<MemorySegment> batchK = randomF32(tail * KV_DIM, 22);
        MemoryView<MemorySegment> batchV = randomF32(tail * KV_DIM, 23);
        MemoryView<MemorySegment> cacheK16 = Views.allocateF16(memory, elements);
        MemoryView<MemorySegment> cacheV16 = Views.allocateF16(memory, elements);
        MemoryView<MemorySegment> cacheK32 = Views.allocateF32(memory, elements);
        MemoryView<MemorySegment> cacheV32 = Views.allocateF32(memory, elements);
        MemoryView<MemorySegment> source = randomF32(2 * elements, 24);
        for (int i = 0; i < elements; i++) {
            roundToF16(source, i, cacheK16, cacheK32, i);
            roundToF16(source, elements + i, cacheV16, cacheV32, i);
        }

        MemoryView<MemorySegment> out16 = Views.allocateF32(memory, tail * QUERY_DIM);
        MemoryView<MemorySegment> out32 = Views.allocateF32(memory, tail * QUERY_DIM);
        prefill(q, out16, cacheK16, cacheV16, batchK, batchV, prefix, tail);
        prefill(q, out32, cacheK32, cacheV32, batchK, batchV, prefix, tail);
        assertClose(out32, out16);
    }

    @Test
    void f16SubnormalValuesSurviveTheRemainderRows() {
        // bidirectional prefill over an F16 batch (the ViT path) with seqLen = 5: four rows go
        // through the 4-row tiles, the fifth through the per-row accumulate with its own decoder.
        // Dim 0 of every V row holds a half subnormal, so dim 0 of every output row is that value
        // whatever the attention weights; a decoder that flushes subnormals returns 0 there.
        int seqLen = 5, elements = seqLen * KV_DIM;
        float tiny = Float.float16ToFloat(Float.floatToFloat16(3e-5f));
        MemoryView<MemorySegment> q = randomF32(seqLen * QUERY_DIM, 31);
        MemoryView<MemorySegment> source = randomF32(2 * elements, 34);
        for (int i = 0; i < elements; i += HEAD_SIZE) setFloat(source, elements + i, tiny);
        MemoryView<MemorySegment> k16 = Views.allocateF16(memory, elements);
        MemoryView<MemorySegment> v16 = Views.allocateF16(memory, elements);
        MemoryView<MemorySegment> k32 = Views.allocateF32(memory, elements);
        MemoryView<MemorySegment> v32 = Views.allocateF32(memory, elements);
        for (int i = 0; i < elements; i++) {
            roundToF16(source, i, k16, k32, i);
            roundToF16(source, elements + i, v16, v32, i);
        }

        MemoryView<MemorySegment> out16 = Views.allocateF32(memory, seqLen * QUERY_DIM);
        MemoryView<MemorySegment> out32 = Views.allocateF32(memory, seqLen * QUERY_DIM);
        FlashAttention.bidirectionalPrefill(
                q, out16, k16, v16, HEADS, seqLen, HEAD_SIZE, KV_DIM, QUERY_DIM, KV_MUL, 1f / 8f);
        FlashAttention.bidirectionalPrefill(
                q, out32, k32, v32, HEADS, seqLen, HEAD_SIZE, KV_DIM, QUERY_DIM, KV_MUL, 1f / 8f);
        for (int i = 0; i < seqLen * QUERY_DIM; i += HEAD_SIZE) {
            assertEquals(tiny, Views.getFloat(out32, i, "out32"), 1e-9f, "reference dim 0 at " + i);
        }
        assertClose(out32, out16);
    }

    @Test
    void wideHeadsMatchPlainCausalAttention() {
        // head sizes that are a multiple of 64 above 64 take the wide 4-row QK tile on 16 lanes
        for (int headSize : new int[] {128, 256}) {
            int seqLen = 37, kvDim = (HEADS / KV_MUL) * headSize, queryDim = HEADS * headSize;
            float scale = (float) (1 / Math.sqrt(headSize));
            float[] q = randomFloats(seqLen * queryDim, 41);
            float[] k = randomFloats(seqLen * kvDim, 42);
            float[] v = randomFloats(seqLen * kvDim, 43);
            MemoryView<MemorySegment> out = Views.allocateF32(memory, seqLen * queryDim);
            FlashAttention.causalPrefill(
                    Views.fromFloatArray(memory, q),
                    out,
                    Views.fromFloatArray(memory, k),
                    Views.fromFloatArray(memory, v),
                    HEADS,
                    0,
                    seqLen,
                    headSize,
                    kvDim,
                    queryDim,
                    KV_MUL,
                    scale);

            for (int t = 0; t < seqLen; t++) {
                for (int h = 0; h < HEADS; h++) {
                    int qAt = t * queryDim + h * headSize, kvHead = (h / KV_MUL) * headSize;
                    double[] weights = new double[t + 1];
                    double max = Double.NEGATIVE_INFINITY, sum = 0;
                    for (int j = 0; j <= t; j++) {
                        double dot = 0;
                        for (int d = 0; d < headSize; d++)
                            dot += (double) q[qAt + d] * k[j * kvDim + kvHead + d];
                        weights[j] = dot * scale;
                        max = Math.max(max, weights[j]);
                    }
                    for (int j = 0; j <= t; j++) sum += weights[j] = Math.exp(weights[j] - max);
                    for (int d = 0; d < headSize; d++) {
                        double expected = 0;
                        for (int j = 0; j <= t; j++)
                            expected += weights[j] / sum * v[j * kvDim + kvHead + d];
                        assertEquals(
                                expected,
                                Views.getFloat(out, qAt + d, "out"),
                                1e-5,
                                "head size " + headSize + ", token " + t + ", head " + h);
                    }
                }
            }
        }
    }

    private static float[] randomFloats(int size, long seed) {
        float[] values = new float[size];
        long state = seed;
        for (int i = 0; i < size; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            values[i] = ((state >>> 40) & 0xffff) / 65536f - 0.5f;
        }
        return values;
    }

    private MemoryView<MemorySegment> randomF32(int size, long seed) {
        return Views.fromFloatArray(memory, randomFloats(size, seed));
    }

    private static MemoryView<MemorySegment> slice(
            MemoryView<MemorySegment> view, long from, long length) {
        return view.slice(0, from, from + length);
    }

    private static void roundToF16(
            MemoryView<MemorySegment> source,
            int sourceIndex,
            MemoryView<MemorySegment> f16,
            MemoryView<MemorySegment> widened,
            int targetIndex) {
        short bits = Float.floatToFloat16(Views.getFloat(source, sourceIndex, "source"));
        f16.memory()
                .base()
                .set(
                        ValueLayout.JAVA_SHORT_UNALIGNED,
                        f16.byteOffset() + (long) targetIndex * Short.BYTES,
                        bits);
        setFloat(widened, targetIndex, Float.float16ToFloat(bits));
    }

    private static void setFloat(MemoryView<MemorySegment> view, int index, float value) {
        view.memory()
                .base()
                .set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED,
                        view.byteOffset() + (long) index * Float.BYTES,
                        value);
    }

    private static void prefill(
            MemoryView<MemorySegment> q,
            MemoryView<MemorySegment> out,
            MemoryView<MemorySegment> cacheK,
            MemoryView<MemorySegment> cacheV,
            MemoryView<MemorySegment> batchK,
            MemoryView<MemorySegment> batchV,
            int start,
            int length) {
        FlashAttention.slidingWindowPrefill(
                q, out, cacheK, cacheV, batchK, batchV, HEADS, start, length, HEAD_SIZE, KV_DIM,
                QUERY_DIM, KV_DIM, KV_MUL, 1f / 8f, 0, 0, null);
    }

    private static void assertClose(
            MemoryView<MemorySegment> expected, MemoryView<MemorySegment> actual) {
        assertEquals(expected.shape().size(), actual.shape().size());
        for (long i = 0; i < expected.shape().size(); i++) {
            float value = Views.getFloat(expected, i, "expected");
            assertEquals(
                    value,
                    Views.getFloat(actual, i, "actual"),
                    Math.max(1e-6f, Math.abs(value) * 1e-5f),
                    "element " + i);
        }
    }
}
