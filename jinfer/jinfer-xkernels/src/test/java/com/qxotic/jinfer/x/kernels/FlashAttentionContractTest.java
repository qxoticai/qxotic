package com.qxotic.jinfer.x.kernels;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
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

    private final PanamaMemoryArena memory = new PanamaMemoryArena(Arena.ofAuto());

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

    private MemoryView<MemorySegment> randomF32(int size, long seed) {
        float[] values = new float[size];
        long state = seed;
        for (int i = 0; i < size; i++) {
            state = state * 6364136223846793005L + 1442695040888963407L;
            values[i] = ((state >>> 40) & 0xffff) / 65536f - 0.5f;
        }
        return Views.fromFloatArray(memory, values);
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
        float value = Float.float16ToFloat(bits);
        if (Math.abs(value) < 0x1p-14f) value = Math.copySign(0f, value);
        widened.memory()
                .base()
                .set(
                        ValueLayout.JAVA_FLOAT_UNALIGNED,
                        widened.byteOffset() + (long) targetIndex * Float.BYTES,
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
