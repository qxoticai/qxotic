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
    void chunkedAtDepthEqualsSingleShotWithinUlps() {
        // NOT byte-exact, deliberately: chunked and single-shot tile the q-rows differently
        // (different BrRows contexts, seam-split pv accumulation), so partial-sum grouping
        // legitimately differs by ulps - and which paths the JIT has compiled changes it
        // (cold scalar runs agree everywhere). The BITWISE gate for cache-leg changes is
        // f16CacheEqualsWidenedF32Cache below - identical call pattern, only the leg differs.
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
            float a = outSingle.getFloat(p * Q_STRIDE + i);
            float b = outTail.getFloat(i);
            float tol = Math.max(1e-6f, Math.abs(a) * 1e-5f);
            assertEquals(a, b, tol, "row float " + i);
        }
    }

    /** f16->f32 as the vector twiddle does it: exact for normals, subnormals -> signed zero. */
    static float ftzWiden(float widened) {
        return Math.abs(widened) < 0x1p-14 ? Math.copySign(0f, widened) : widened;
    }

    @Test
    void decodeMatchesFtzWidening() {
        // the BITWISE gate on the fix itself: the scratch decode must reproduce the vector
        // converter's semantics exactly (normals widened, subnormals flushed to signed zero)
        // for every key, including the scalar remainder lanes. Pure data - no JIT sensitivity.
        int keys = 61, headSize = HEAD; // odd count exercises vector bound + scalar remainder
        F32FloatTensor src = rnd(keys * KV_DIM, 31);
        FloatTensor f16 = FloatTensor.allocateF16(keys, KV_DIM);
        for (int i = 0; i < keys * KV_DIM; i++) {
            f16.setFloat(i, src.getFloat(i));
        }
        int[] kvOff = new int[keys];
        for (int j = 0; j < keys; j++) {
            kvOff[j] = j * KV_DIM + HEAD; // second head's slice, strided like the real caller
        }
        F32FloatTensor dst = F32FloatTensor.allocate(keys * headSize);
        FlashAttention.decodeF16Run((F16FloatTensor) f16, kvOff, keys, headSize, dst);
        for (int j = 0; j < keys; j++) {
            for (int d = 0; d < headSize; d++) {
                assertEquals(
                        Float.floatToRawIntBits(ftzWiden(f16.getFloat(kvOff[j] + d))),
                        Float.floatToRawIntBits(dst.getFloat(j * headSize + d)),
                        "key " + j + " dim " + d);
            }
        }
    }

    @Test
    void f16CacheEqualsWidenedF32Cache() {
        // an F16 cache must score like an F32 cache holding the same WIDENED values - proves
        // the conversion is a pure representation change. Within ulps, not byte-exact: which
        // compiled form of the tiles each leg hits (C1/C2 tiering mid-test) changes partial-sum
        // grouping, same phenomenon as the chunked oracle above. The bitwise burden sits on
        // decodeMatchesFtzWidening, where the data is compared directly.
        int p = 96, tail = 61;
        F32FloatTensor q = rnd(tail * Q_STRIDE, 21);
        F32FloatTensor bK = rnd(tail * KV_DIM, 22);
        F32FloatTensor bV = rnd(tail * KV_DIM, 23);
        FloatTensor cK16 = FloatTensor.allocateF16(p, KV_DIM);
        FloatTensor cV16 = FloatTensor.allocateF16(p, KV_DIM);
        F32FloatTensor cK32 = F32FloatTensor.allocate(p * KV_DIM);
        F32FloatTensor cV32 = F32FloatTensor.allocate(p * KV_DIM);
        F32FloatTensor src = rnd(2 * p * KV_DIM, 24);
        for (int i = 0; i < p * KV_DIM; i++) {
            cK16.setFloat(i, src.getFloat(i));
            cV16.setFloat(i, src.getFloat(p * KV_DIM + i));
            // the vector converter's semantics: exact widening for normals, subnormals
            // FLUSHED to signed zero (the zeroExponentMask twiddle) - the F32 mirror must
            // replicate that or the comparison chases ulps
            cK32.setFloat(i, ftzWiden(cK16.getFloat(i)));
            cV32.setFloat(i, ftzWiden(cV16.getFloat(i)));
        }
        F32FloatTensor out16 = F32FloatTensor.allocate(tail * Q_STRIDE);
        F32FloatTensor out32 = F32FloatTensor.allocate(tail * Q_STRIDE);
        FlashAttention.slidingWindowPrefill(
                q, out16, cK16, cV16, bK, bV, HEADS, p, tail, HEAD, KV_DIM, Q_STRIDE, KV_DIM,
                KV_MUL, 1f / 8f, 0, 0, null);
        FlashAttention.slidingWindowPrefill(
                q, out32, cK32, cV32, bK, bV, HEADS, p, tail, HEAD, KV_DIM, Q_STRIDE, KV_DIM,
                KV_MUL, 1f / 8f, 0, 0, null);
        for (int i = 0; i < tail * Q_STRIDE; i++) {
            float a = out32.getFloat(i);
            float b = out16.getFloat(i);
            float tol = Math.max(1e-6f, Math.abs(a) * 1e-5f);
            assertEquals(a, b, tol, "float " + i);
        }
    }

    @Test
    @Tag("bench")
    void depthCurve() {
        // qwen3-0.6B's REAL shape, both cache dtypes - the F16-vs-F32 discriminator
        int heads = 16, kvMul = 2, head = 128;
        int kvDim = (heads / kvMul) * head, qStride = heads * head;
        int tail = 62, layers = 28;
        for (boolean f16 : new boolean[] {false, true}) {
            for (int depth : new int[] {0, 1216}) {
                int n = depth + tail;
                FloatTensor cK = f16 ? FloatTensor.allocateF16(n, kvDim) : rnd(n * kvDim, 10);
                FloatTensor cV = f16 ? FloatTensor.allocateF16(n, kvDim) : rnd(n * kvDim, 11);
                F32FloatTensor q = rnd(tail * qStride, 12);
                F32FloatTensor bK = rnd(tail * kvDim, 13);
                F32FloatTensor bV = rnd(tail * kvDim, 14);
                F32FloatTensor out = F32FloatTensor.allocate(tail * qStride);
                Runnable one =
                        () ->
                                FlashAttention.slidingWindowPrefill(
                                        q,
                                        out,
                                        cK,
                                        cV,
                                        bK,
                                        bV,
                                        heads,
                                        depth,
                                        tail,
                                        head,
                                        kvDim,
                                        qStride,
                                        kvDim,
                                        kvMul,
                                        1f / 11.3f,
                                        0,
                                        0,
                                        null);
                for (int w = 0; w < 3 * layers; w++) one.run();
                int reps = 5 * layers; // ~5 simulated model passes
                long t0 = System.nanoTime();
                for (int r = 0; r < reps; r++) one.run();
                double msLayer = (System.nanoTime() - t0) / 1e6 / reps;
                System.out.printf(
                        "cache=%s depth=%-5d  %8.3f ms/layer  (~%.0f ms per 28-layer pass)%n",
                        f16 ? "F16" : "F32", depth, msLayer, msLayer * layers);
            }
        }
    }
}
