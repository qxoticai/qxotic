package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The net under any cached-prefix kernel work. Oracle: chunked prefill at depth must produce
 * BYTE-identical outputs to a single-shot prefill of the concatenation (per-row math is independent
 * and both process kv blocks in the same ascending order - the cache/batch seam only splits tile
 * calls, never reorders keys). Bench: the depth curve that exposed the ~20x cache-leg tax,
 * reproduced kernel-only.
 */
class FlashPrefillDepthTest {

    static final int HEADS = 4, KV_MUL = 2, HEAD = 64;
    static final int KV_DIM = (HEADS / KV_MUL) * HEAD;
    static final int Q_STRIDE = HEADS * HEAD;

    static F32FloatTensor rnd(int size, long seed) {
        F32FloatTensor t = F32FloatTensor.allocate(size);
        long x = seed;
        for (int i = 0; i < size; i++) {
            x = x * 6364136223846793005L + 1442695040888963407L;
            t.setFloat(i, ((x >>> 40) & 0xFFFF) / 65536f - 0.5f);
        }
        return t;
    }

    static void prefill(
            F32FloatTensor q,
            F32FloatTensor out,
            F32FloatTensor cK,
            F32FloatTensor cV,
            F32FloatTensor bK,
            F32FloatTensor bV,
            int startPos,
            int seqLen) {
        FlashAttention.slidingWindowPrefill(
                q, out, cK, cV, bK, bV, HEADS, startPos, seqLen, HEAD, KV_DIM, Q_STRIDE, KV_DIM,
                KV_MUL, 1f / 8f, 0, 0, null);
    }

    @Test
    void chunkedAtDepthEqualsSingleShotByteExact() {
        int p = 96, tail = 61, n = p + tail; // 61: exercises the QT remainder path too
        F32FloatTensor q = rnd(n * Q_STRIDE, 1);
        F32FloatTensor k = rnd(n * KV_DIM, 2);
        F32FloatTensor v = rnd(n * KV_DIM, 3);
        F32FloatTensor empty = F32FloatTensor.allocate(1);

        F32FloatTensor outSingle = F32FloatTensor.allocate(n * Q_STRIDE);
        prefill(q, outSingle, empty, empty, k, v, 0, n);

        // chunked: prefix chunk at 0, then the tail attends prefix-from-cache + itself-from-batch
        F32FloatTensor outPrefix = F32FloatTensor.allocate(n * Q_STRIDE);
        prefill(q, outPrefix, empty, empty, k, v, 0, p);
        F32FloatTensor qTail = F32FloatTensor.allocate(tail * Q_STRIDE);
        F32FloatTensor kTail = F32FloatTensor.allocate(tail * KV_DIM);
        F32FloatTensor vTail = F32FloatTensor.allocate(tail * KV_DIM);
        q.copyTo((long) p * Q_STRIDE, qTail, 0, tail * Q_STRIDE);
        k.copyTo((long) p * KV_DIM, kTail, 0, tail * KV_DIM);
        v.copyTo((long) p * KV_DIM, vTail, 0, tail * KV_DIM);
        F32FloatTensor outTail = F32FloatTensor.allocate(tail * Q_STRIDE);
        prefill(qTail, outTail, k, v, kTail, vTail, p, tail);

        for (int i = 0; i < tail * Q_STRIDE; i++) {
            assertEquals(
                    Float.floatToRawIntBits(outSingle.getFloat(p * Q_STRIDE + i)),
                    Float.floatToRawIntBits(outTail.getFloat(i)),
                    "row float " + i);
        }
    }

    @Test
    @Tag("bench")
    void depthCurve() {
        int tail = 62;
        for (int depth : new int[] {0, 256, 1024, 4096}) {
            int n = depth + tail;
            F32FloatTensor cK = rnd(n * KV_DIM, 10);
            F32FloatTensor cV = rnd(n * KV_DIM, 11);
            F32FloatTensor q = rnd(tail * Q_STRIDE, 12);
            F32FloatTensor bK = rnd(tail * KV_DIM, 13);
            F32FloatTensor bV = rnd(tail * KV_DIM, 14);
            F32FloatTensor out = F32FloatTensor.allocate(tail * Q_STRIDE);
            for (int w = 0; w < 3; w++) prefill(q, out, cK, cV, bK, bV, depth, tail);
            int reps = 50;
            long t0 = System.nanoTime();
            for (int r = 0; r < reps; r++) prefill(q, out, cK, cV, bK, bV, depth, tail);
            double ms = (System.nanoTime() - t0) / 1e6 / reps;
            double gflop = 2.0 * 2 * HEADS * tail * (depth + tail / 2.0) * HEAD / 1e9;
            System.out.printf(
                    "depth=%-5d tail=%d  %8.3f ms   %6.2f GFLOP/s%n",
                    depth, tail, ms, gflop / (ms / 1000));
        }
    }
}
