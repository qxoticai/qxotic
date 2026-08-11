package com.qxotic.jinfer.x.kernels;

import com.qxotic.format.gguf.GGMLType;
import com.qxotic.jinfer.F32FloatTensor;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jota.DataType;
import com.qxotic.jota.Shape;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential oracles: ported x.FlashAttention vs jinfer-core FlashAttention on identical inputs —
 * causal/sliding-window/bidirectional prefill and flash decode, over F32 and F16 KV sources, with
 * and without sinks, exercising the tiled, partial-row and rolling paths. Ulp-bound (Vector API
 * tier nondeterminism — see Oracles).
 */
class FlashAttentionTest {

    private static final int N_HEADS = 4;
    private static final int HEAD_SIZE = 64;
    private static final int KV_MUL = 2;
    private static final int KV_DIM = (N_HEADS / KV_MUL) * HEAD_SIZE; // 128
    private static final int QUERY_DIM = N_HEADS * HEAD_SIZE; // 256
    private static final float SCALE = 1.0f / (float) Math.sqrt(HEAD_SIZE);

    private final Arena arena = Arena.ofAuto();

    // ---- plumbing -------------------------------------------------------------

    private MemorySegment randomF32(int n, long seed) {
        return Oracles.f32(arena, n, seed);
    }

    /**
     * F16 segment holding the same values as {@code src} (round-tripped through floatToFloat16).
     */
    private MemorySegment asF16(MemorySegment src, int n) {
        MemorySegment seg = arena.allocate(2L * n, 64);
        for (int i = 0; i < n; i++) {
            seg.set(
                    ValueLayout.JAVA_SHORT_UNALIGNED,
                    2L * i,
                    Float.floatToFloat16(src.get(ValueLayout.JAVA_FLOAT_UNALIGNED, 4L * i)));
        }
        return seg;
    }

    private FloatTensor oldTensor(MemorySegment seg, int n, boolean f16) {
        return FloatTensor.create(f16 ? GGMLType.F16 : GGMLType.F32, n, seg);
    }

    private MemoryView<MemorySegment> newView(MemorySegment seg, int n, boolean f16) {
        return Views.wrap(seg, f16 ? DataType.FP16 : DataType.FP32, Shape.flat((long) n));
    }

    private MemorySegment copyOf(MemorySegment seg) {
        MemorySegment copy = arena.allocate(seg.byteSize(), 64);
        MemorySegment.copy(seg, 0, copy, 0, seg.byteSize());
        return copy;
    }

    private void assertOutClose(MemorySegment expected, MemorySegment actual, int n, String what) {
        Oracles.assertClose(expected, actual, n, what, 1e-4);
    }

    // ---- causal prefill ---------------------------------------------------------

    private void causalPrefillParity(boolean f16Cache) {
        int startPos = 5, seqLen = 37;
        int cacheN = (startPos + seqLen) * KV_DIM;
        int qN = seqLen * QUERY_DIM;
        MemorySegment cacheK = randomF32(cacheN, 1), cacheV = randomF32(cacheN, 2);
        MemorySegment q = randomF32(qN, 3);
        MemorySegment outOld = arena.allocate(4L * qN, 64);
        MemorySegment outNew = copyOf(outOld);

        MemorySegment ckSeg = f16Cache ? asF16(cacheK, cacheN) : cacheK;
        MemorySegment cvSeg = f16Cache ? asF16(cacheV, cacheN) : cacheV;
        com.qxotic.jinfer.FlashAttention.causalPrefill(
                (F32FloatTensor) oldTensor(q, qN, false),
                (F32FloatTensor) oldTensor(outOld, qN, false),
                oldTensor(ckSeg, cacheN, f16Cache),
                oldTensor(cvSeg, cacheN, f16Cache),
                N_HEADS,
                startPos,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_MUL);
        FlashAttention.causalPrefill(
                newView(q, qN, false),
                newView(outNew, qN, false),
                newView(ckSeg, cacheN, f16Cache),
                newView(cvSeg, cacheN, f16Cache),
                N_HEADS,
                startPos,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_MUL);
        assertOutClose(outOld, outNew, qN, "causalPrefill f16=" + f16Cache);
    }

    @Test
    void causalPrefillParityF32Cache() {
        causalPrefillParity(false);
    }

    @Test
    void causalPrefillParityF16Cache() {
        causalPrefillParity(true);
    }

    // ---- sliding-window prefill ---------------------------------------------------

    private void slidingWindowParity(
            int window, int ringMask, boolean withSinks, boolean f16Cache, long seed) {
        int startPos = 70, seqLen = 13;
        int cacheCap = 128, cacheN = cacheCap * KV_DIM;
        int batchN = seqLen * KV_DIM;
        int qN = seqLen * QUERY_DIM;
        Random rng = new Random(seed);
        MemorySegment cacheK = randomF32(cacheN, seed), cacheV = randomF32(cacheN, seed + 1);
        MemorySegment batchK = randomF32(batchN, seed + 2), batchV = randomF32(batchN, seed + 3);
        MemorySegment q = randomF32(qN, seed + 4);
        MemorySegment sinks = withSinks ? randomF32(N_HEADS, seed + 5) : null;
        MemorySegment outOld = arena.allocate(4L * qN, 64);
        MemorySegment outNew = copyOf(outOld);

        MemorySegment ckSeg = f16Cache ? asF16(cacheK, cacheN) : cacheK;
        MemorySegment cvSeg = f16Cache ? asF16(cacheV, cacheN) : cacheV;
        FloatTensor sinksOld = withSinks ? oldTensor(sinks, N_HEADS, false) : null;
        MemoryView<MemorySegment> sinksNew = withSinks ? newView(sinks, N_HEADS, false) : null;
        com.qxotic.jinfer.FlashAttention.slidingWindowPrefill(
                oldTensor(q, qN, false),
                oldTensor(outOld, qN, false),
                oldTensor(ckSeg, cacheN, f16Cache),
                oldTensor(cvSeg, cacheN, f16Cache),
                oldTensor(batchK, batchN, false),
                oldTensor(batchV, batchN, false),
                N_HEADS,
                startPos,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_DIM,
                KV_MUL,
                SCALE,
                window,
                ringMask,
                sinksOld);
        FlashAttention.slidingWindowPrefill(
                newView(q, qN, false),
                newView(outNew, qN, false),
                newView(ckSeg, cacheN, f16Cache),
                newView(cvSeg, cacheN, f16Cache),
                newView(batchK, batchN, false),
                newView(batchV, batchN, false),
                N_HEADS,
                startPos,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_DIM,
                KV_MUL,
                SCALE,
                window,
                ringMask,
                sinksNew);
        assertOutClose(
                outOld,
                outNew,
                qN,
                "slidingWindow window=" + window + " sinks=" + withSinks + " f16=" + f16Cache);
    }

    @Test
    void slidingWindowRingF16Cache() {
        slidingWindowParity(16, 31, false, true, 10);
    }

    @Test
    void slidingWindowRingWithSinks() {
        slidingWindowParity(16, 31, true, true, 20);
    }

    @Test
    void slidingWindowFullF32Cache() {
        slidingWindowParity(0, 0, false, false, 30);
    }

    // ---- bidirectional prefill -----------------------------------------------------

    @Test
    void bidirectionalPrefillParity() {
        int seqLen = 17;
        int batchN = seqLen * KV_DIM;
        int qN = seqLen * QUERY_DIM;
        MemorySegment batchK = randomF32(batchN, 40), batchV = randomF32(batchN, 41);
        MemorySegment q = randomF32(qN, 42);
        MemorySegment outOld = arena.allocate(4L * qN, 64);
        MemorySegment outNew = copyOf(outOld);
        MemorySegment bK16 = asF16(batchK, batchN);

        com.qxotic.jinfer.FlashAttention.bidirectionalPrefill(
                oldTensor(q, qN, false),
                oldTensor(outOld, qN, false),
                oldTensor(bK16, batchN, true),
                oldTensor(batchV, batchN, false),
                N_HEADS,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_MUL,
                SCALE);
        FlashAttention.bidirectionalPrefill(
                newView(q, qN, false),
                newView(outNew, qN, false),
                newView(bK16, batchN, true),
                newView(batchV, batchN, false),
                N_HEADS,
                seqLen,
                HEAD_SIZE,
                KV_DIM,
                QUERY_DIM,
                KV_MUL,
                SCALE);
        assertOutClose(outOld, outNew, qN, "bidirectionalPrefill");
    }

    // ---- flash decode ----------------------------------------------------------------

    private void flashDecodeParity(
            int position, boolean f16Cache, boolean withBatch, boolean withSinks, long seed) {
        int cacheCap = 1024, cacheN = cacheCap * KV_DIM;
        int qN = QUERY_DIM;
        MemorySegment cacheK = randomF32(cacheN, seed), cacheV = randomF32(cacheN, seed + 1);
        MemorySegment batchK = withBatch ? randomF32(KV_DIM, seed + 2) : null;
        MemorySegment batchV = withBatch ? randomF32(KV_DIM, seed + 3) : null;
        MemorySegment q = randomF32(qN, seed + 4);
        MemorySegment sinks = withSinks ? randomF32(N_HEADS, seed + 5) : null;
        MemorySegment outOld = arena.allocate(4L * qN, 64);
        MemorySegment outNew = copyOf(outOld);

        MemorySegment ckSeg = f16Cache ? asF16(cacheK, cacheN) : cacheK;
        MemorySegment cvSeg = f16Cache ? asF16(cacheV, cacheN) : cacheV;
        com.qxotic.jinfer.FlashAttention.flashDecode(
                (F32FloatTensor) oldTensor(q, qN, false),
                (F32FloatTensor) oldTensor(outOld, qN, false),
                oldTensor(ckSeg, cacheN, f16Cache),
                oldTensor(cvSeg, cacheN, f16Cache),
                withBatch ? oldTensor(batchK, KV_DIM, false) : null,
                withBatch ? oldTensor(batchV, KV_DIM, false) : null,
                N_HEADS,
                position,
                0,
                HEAD_SIZE,
                KV_DIM,
                KV_MUL,
                SCALE,
                0,
                withSinks ? oldTensor(sinks, N_HEADS, false) : null,
                new com.qxotic.jinfer.FlashAttention.DecodeScratch(arena));
        FlashAttention.flashDecode(
                newView(q, qN, false),
                newView(outNew, qN, false),
                newView(ckSeg, cacheN, f16Cache),
                newView(cvSeg, cacheN, f16Cache),
                withBatch ? newView(batchK, KV_DIM, false) : null,
                withBatch ? newView(batchV, KV_DIM, false) : null,
                N_HEADS,
                position,
                0,
                HEAD_SIZE,
                KV_DIM,
                KV_MUL,
                SCALE,
                0,
                withSinks ? newView(sinks, N_HEADS, false) : null,
                new FlashAttention.DecodeScratch(new PanamaMemoryArena(arena)));
        assertOutClose(
                outOld,
                outNew,
                qN,
                "flashDecode pos=" + position + " f16=" + f16Cache + " batch=" + withBatch);
    }

    @Test
    void flashDecodePartitionedF16Cache() {
        // range 601 > DECODE_BLOCK_SIZE(512): exercises the partition+merge path
        flashDecodeParity(600, true, false, false, 50);
    }

    @Test
    void flashDecodeRollingWithBatchAndSinks() {
        flashDecodeParity(37, true, true, true, 60);
    }

    @Test
    void flashDecodeRollingF32Cache() {
        flashDecodeParity(100, false, true, false, 70);
    }
}
