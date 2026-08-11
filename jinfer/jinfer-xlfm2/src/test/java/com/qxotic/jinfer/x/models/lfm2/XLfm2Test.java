package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.x.Segments;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * THE cycle-1 gate: the x LFM2.5 port vs the old FloatTensor LFM2 port on the REAL
 * LFM2.5-2.6B-Q8_0.gguf, ONE shared tokenizer instance fed to both loaders, greedy N=64 — strict
 * token-id parity at every step plus a logits max-abs-diff check on the prefill and final steps
 * (both trees route gemm/gemv to the same dot-based Java floor: the JAM backends are
 * surefire-excluded). Skipped when the model is not in the HF cache.
 */
class XLfm2Test {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");
    private static final int N_TOKENS = 64;
    private static final float LOGIT_TOLERANCE = 1e-2f;

    private static Path model;

    @BeforeAll
    static void findModel() throws IOException {
        Path repo = HF_CACHE.resolve("models--LiquidAI--LFM2.5-2.6B-GGUF/snapshots");
        if (Files.isDirectory(repo)) {
            try (Stream<Path> snaps = Files.list(repo)) {
                model =
                        snaps.flatMap(
                                        s -> {
                                            try {
                                                return Files.list(s);
                                            } catch (IOException e) {
                                                return Stream.empty();
                                            }
                                        })
                                .filter(
                                        p ->
                                                p.getFileName()
                                                        .toString()
                                                        .equals("LFM2.5-2.6B-Q8_0.gguf"))
                                .findFirst()
                                .orElse(null);
            }
        }
    }

    @Test
    void greedyParity() throws Exception {
        assumeTrue(model != null, "LFM2.5-2.6B-Q8_0.gguf not in the HF cache");
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);

            List<Integer> prompt = new ArrayList<>();
            prompt.add(SpecialTokens.find(tokenizer, "<bos>").orElse(1));
            for (int id : tokenizer.encodeToArray("The capital of France is")) prompt.add(id);
            int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();

            int[] oldIds;
            float[] oldFirst, oldLast;
            var om =
                    com.qxotic.jinfer.models.lfm2.Lfm2.loadModel(
                            channel, gguf, Arena.ofAuto(), tokenizer);
            var oc = om.config();
            int vocab = oc.vocabularySize();
            try (var os =
                    om.newState(
                            Math.min(oc.contextLength(), ids.length + N_TOKENS + 16),
                            Math.max(16, ids.length))) {
                om.ingest(os, com.qxotic.jinfer.Batch.prefill(ids));
                oldIds = new int[N_TOKENS];
                FloatTensor logits = om.logits(os);
                oldFirst = snapshotOld(logits, vocab);
                int tok = logits.argmax();
                for (int n = 0; n < N_TOKENS; n++) {
                    oldIds[n] = tok;
                    om.ingest(os, com.qxotic.jinfer.Batch.step(tok));
                    logits = om.logits(os);
                    tok = logits.argmax();
                }
                oldLast = snapshotOld(logits, vocab);
            }

            int[] xIds;
            float[] xFirst, xLast;
            var xm = Lfm2.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            var xc = xm.config();
            try (var xs =
                    xm.newState(
                            Math.min(xc.contextLength(), ids.length + N_TOKENS + 16),
                            Math.max(16, ids.length))) {
                xm.ingest(xs, Batch.prefill(ids));
                xIds = new int[N_TOKENS];
                MemoryView<MemorySegment> logits =
                        Views.castToSegmentBacked(xm.logits(xs), "logits");
                xFirst = snapshotX(logits, vocab);
                int tok = Ops.argmax(logits, 0, vocab);
                for (int n = 0; n < N_TOKENS; n++) {
                    xIds[n] = tok;
                    xm.ingest(xs, Batch.step(tok));
                    logits = Views.castToSegmentBacked(xm.logits(xs), "logits");
                    tok = Ops.argmax(logits, 0, vocab);
                }
                xLast = snapshotX(logits, vocab);
            }

            float firstDiff = maxAbsDiff(oldFirst, xFirst);
            float lastDiff = maxAbsDiff(oldLast, xLast);
            System.err.printf(
                    "logits max-abs-diff: prefill=%.6g final=%.6g%n", firstDiff, lastDiff);
            assertTrue(firstDiff < LOGIT_TOLERANCE, "prefill logits diverged: " + firstDiff);
            assertTrue(lastDiff < LOGIT_TOLERANCE, "final logits diverged: " + lastDiff);
            assertArrayEquals(oldIds, xIds, "greedy token-id divergence");
        }
    }

    private static float[] snapshotOld(FloatTensor logits, int vocab) {
        float[] out = new float[vocab];
        for (int i = 0; i < vocab; i++) out[i] = logits.getFloat(i);
        return out;
    }

    private static float[] snapshotX(MemoryView<MemorySegment> logits, int vocab) {
        Views.Raw r = Views.rawF32(logits, "logits");
        float[] out = new float[vocab];
        for (int i = 0; i < vocab; i++)
            out[i] = Segments.readFloat(r.vseg(), r.vbase() + (long) i * Float.BYTES);
        return out;
    }

    private static float maxAbsDiff(float[] a, float[] b) {
        float max = 0;
        for (int i = 0; i < a.length; i++) max = Math.max(max, Math.abs(a[i] - b[i]));
        return max;
    }
}
