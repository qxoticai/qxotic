package com.qxotic.jinfer.x.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * THE cycle-2 gate: the x Qwen3 port vs the old FloatTensor Qwen3 port on the REAL checkpoints,
 * packed ragged {@code EmbeddingModel.embed} with a deliberately small batchCapacity so sequences
 * span chunks (exercising the KV-carry the boundary's causal streaming law rests on) — per-sequence
 * cosine + max-abs-diff on the L2-normalized vectors. Both trees route gemm/gemv to the same
 * dot-based Java floor (the JAM backends are surefire-excluded). Two legs: Qwen3-Embedding-0.6B
 * (the embedder) and Qwen3-Reranker-0.6B (the judge — same backbone, the tied-head verdict itself
 * is gated by the Qwen3Reranker port's test). Skipped per leg when the checkpoint is not cached.
 */
class XQwen3Test {

    private static final int BATCH_CAPACITY =
            Integer.getInteger("xqwen3.batchCapacity", 16); // < longest sequence: forces chunks
    private static final float MAX_ABS_TOLERANCE = 1e-2f;
    private static final double COSINE_TOLERANCE = 1e-4;

    @Test
    void embedParityEmbeddingCheckpoint() throws Exception {
        assertEmbedParity(TestModels.require("hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"));
    }

    @Test
    void embedParityRerankerCheckpoint() throws Exception {
        assertEmbedParity(TestModels.require("hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0"));
    }

    private static final String[] TEXTS = {
        "The capital of France is Paris.",
        "Qwen3 embedding models pool the final hidden state of the last token and L2-normalize it"
            + " before the vector leaves the model, which makes cosine similarity a dot product.",
        "short",
        "A packed ragged batch streams several sequences through one causal context; a sequence"
            + " longer than the batch capacity spans chunk boundaries and its keys and values are"
            + " carried forward so attention still sees the whole prefix."
    };

    private static void assertEmbedParity(Path model) throws Exception {
        var om = com.qxotic.jinfer.models.qwen3.Qwen3.loadModel(model, Arena.ofAuto());
        var xm = Qwen3.loadModel(model, Arena.ofAuto());

        int[][] idsPerSeq = new int[TEXTS.length][];
        int total = 0;
        for (int i = 0; i < TEXTS.length; i++) {
            idsPerSeq[i] = om.tokenizer().encodeToArray(TEXTS[i]);
            total += idsPerSeq[i].length;
        }
        int[] ids = new int[total];
        int[] seqLen = new int[TEXTS.length];
        int at = 0;
        for (int i = 0; i < TEXTS.length; i++) {
            System.arraycopy(idsPerSeq[i], 0, ids, at, idsPerSeq[i].length);
            seqLen[i] = idsPerSeq[i].length;
            at += seqLen[i];
        }

        int dim = om.config().embeddingLength();
        int ctx = Math.min(om.config().contextLength(), total + 16);

        float[][] oldVecs;
        try (var os = om.newState(ctx, BATCH_CAPACITY)) {
            List<float[]> out = new ArrayList<>();
            om.embed(
                    os,
                    new com.qxotic.jinfer.Batch.Input.Sequences(
                            new com.qxotic.jinfer.Batch.Input.Tokens(ids), seqLen),
                    e -> out.add(snapshotOld(e, dim)));
            oldVecs = out.toArray(new float[0][]);
        }

        float[][] xVecs;
        try (var xs = xm.newState(ctx, BATCH_CAPACITY)) {
            List<float[]> out = new ArrayList<>();
            xm.embedAll(
                    xs,
                    new Batch.Input.Sequences(new Batch.Input.Tokens(ids), seqLen),
                    e -> out.add(snapshotX(e, dim)));
            xVecs = out.toArray(new float[0][]);
        }

        assertTrue(oldVecs.length == xVecs.length, "sequence count mismatch");
        for (int i = 0; i < oldVecs.length; i++) {
            float diff = maxAbsDiff(oldVecs[i], xVecs[i]);
            double cos = cosine(oldVecs[i], xVecs[i]);
            System.err.printf(
                    "seq %d (len %3d): max-abs-diff=%.6g cosine=%.8f%n", i, seqLen[i], diff, cos);
            assertTrue(
                    diff < MAX_ABS_TOLERANCE,
                    "seq " + i + " embeddings diverged: max-abs-diff " + diff);
            assertTrue(
                    cos > 1.0 - COSINE_TOLERANCE,
                    "seq " + i + " embeddings diverged: cosine " + cos);
        }
    }

    private static float[] snapshotOld(FloatTensor e, int dim) {
        float[] out = new float[dim];
        for (int i = 0; i < dim; i++) out[i] = e.getFloat(i);
        return out;
    }

    private static float[] snapshotX(MemoryView<?> e, int dim) {
        MemoryView<MemorySegment> view = Views.castToSegmentBacked(e, "embedding");
        float[] out = new float[dim];
        for (int i = 0; i < dim; i++) out[i] = Views.getFloat(view, i, "embedding");
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
        return max;
    }

    private static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }
}
