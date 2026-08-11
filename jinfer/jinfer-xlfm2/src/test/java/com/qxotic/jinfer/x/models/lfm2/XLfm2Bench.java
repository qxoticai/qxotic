package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.llm.SpecialTokens;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Cycle-1 perf gate: decode tok/s A/B of the x port vs the old FloatTensor LFM2 on the REAL
 * LFM2.5-2.6B-Q8_0.gguf, same JVM, same dot-based Java floor (JAM surefire-excluded), warmup +
 * timed rounds, REPORT ONLY (numbers go to the plan; "within noise or faster" is judged by a human,
 * not asserted — CI machines are too noisy for a hard ratio).
 */
class XLfm2Bench {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");
    private static final int WARMUP = 32;
    private static final int TIMED = 128;

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
    @Tag("bench")
    void ab() throws Exception {
        assumeTrue(model != null, "LFM2.5-2.6B-Q8_0.gguf not in the HF cache");
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            List<Integer> prompt = new ArrayList<>();
            prompt.add(SpecialTokens.find(tokenizer, "<bos>").orElse(1));
            for (int id : tokenizer.encodeToArray("The capital of France is")) prompt.add(id);
            int[] ids = prompt.stream().mapToInt(Integer::intValue).toArray();

            double oldTps = benchOld(channel, gguf, tokenizer, ids);
            double xTps = benchX(channel, gguf, tokenizer, ids);
            System.err.printf(
                    "decode tok/s: old=%.2f x=%.2f ratio=%.3f%n", oldTps, xTps, xTps / oldTps);
        }
    }

    private static double benchOld(FileChannel channel, GGUF gguf, Tokenizer tk, int[] ids)
            throws Exception {
        var m = com.qxotic.jinfer.models.lfm2.Lfm2.loadModel(channel, gguf, Arena.ofAuto(), tk);
        var c = m.config();
        try (var s =
                m.newState(
                        Math.min(c.contextLength(), ids.length + WARMUP + TIMED + 16),
                        Math.max(16, ids.length))) {
            m.ingest(s, com.qxotic.jinfer.Batch.prefill(ids));
            int tok = m.logits(s).argmax();
            for (int n = 0; n < WARMUP; n++) {
                m.ingest(s, com.qxotic.jinfer.Batch.step(tok));
                tok = m.logits(s).argmax();
            }
            long t0 = System.nanoTime();
            for (int n = 0; n < TIMED; n++) {
                m.ingest(s, com.qxotic.jinfer.Batch.step(tok));
                tok = m.logits(s).argmax();
            }
            return TIMED / ((System.nanoTime() - t0) / 1e9);
        }
    }

    private static double benchX(FileChannel channel, GGUF gguf, Tokenizer tk, int[] ids)
            throws Exception {
        var m = Lfm2.loadModel(channel, gguf, Arena.ofAuto(), tk);
        var c = m.config();
        try (var s =
                m.newState(
                        Math.min(c.contextLength(), ids.length + WARMUP + TIMED + 16),
                        Math.max(16, ids.length))) {
            m.ingest(s, Batch.prefill(ids));
            int tok =
                    Ops.argmax(
                            Views.castToSegmentBacked(m.logits(s), "logits"),
                            0,
                            c.vocabularySize());
            for (int n = 0; n < WARMUP; n++) {
                m.ingest(s, Batch.step(tok));
                tok =
                        Ops.argmax(
                                Views.castToSegmentBacked(m.logits(s), "logits"),
                                0,
                                c.vocabularySize());
            }
            long t0 = System.nanoTime();
            for (int n = 0; n < TIMED; n++) {
                m.ingest(s, Batch.step(tok));
                tok =
                        Ops.argmax(
                                Views.castToSegmentBacked(m.logits(s), "logits"),
                                0,
                                c.vocabularySize());
            }
            return TIMED / ((System.nanoTime() - t0) / 1e9);
        }
    }
}
