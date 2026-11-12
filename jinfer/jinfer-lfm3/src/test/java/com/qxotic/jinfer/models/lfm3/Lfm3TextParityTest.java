package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The backbone against llama.cpp on the SAME F32 weights (BF16 widened exactly) with an F32 KV
 * cache and no flash attention on the reference side, over a 628-token prompt that crosses the
 * 256-token sliding window: every layer's residual and the logits of every position.
 */
@Tag("integration")
final class Lfm3TextParityTest {

    static final String F32 = "hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-F32.gguf";

    @Test
    void everyLayerAndEveryPositionMatchLlamaCpp() throws Exception {
        Path dir = Oracle.dir("text-f32");
        int[] tokens = Oracle.ints(dir.resolve("tokens.i32"));
        float[] expectedLogits = Oracle.floats(dir.resolve("logits.f32"));
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model = Lfm3.loadModel(TestModels.require(F32), arena);
            int dim = model.configuration().embeddingLength();
            int vocab = model.configuration().vocabularySize();
            int n = tokens.length;
            try (Lfm3.State state = model.newState(n, n)) {
                double[] worst = {0};
                model.ingest(
                        state,
                        Batch.score(tokens),
                        layer -> {
                            try {
                                Oracle.Tensor ref = Oracle.tensor(dir, "l_out-" + layer);
                                Oracle.Score s =
                                        Oracle.score(
                                                ref.data(),
                                                Oracle.rows(state.residual, n, dim),
                                                dim);
                                System.out.printf("l_out-%-2d %s%n", layer, s);
                                worst[0] = Math.max(worst[0], s.maxRel());
                            } catch (IOException e) {
                                throw new UncheckedIOException(e);
                            }
                        });
                float[] logits = new float[n * vocab];
                for (int i = 0; i < n; i++)
                    Views.copyToArray(
                            Views.castToSegmentBacked(model.logits(state, i), "logits"),
                            0,
                            logits,
                            i * vocab,
                            vocab,
                            "logits");
                double worstKl = 0;
                for (int i = 0; i < n; i++) {
                    worstKl =
                            Math.max(
                                    worstKl,
                                    Oracle.kl(expectedLogits, i * vocab, logits, i * vocab, vocab));
                    // a different pick must be a tie in the reference
                    int want = Oracle.argmax(expectedLogits, i * vocab, vocab);
                    int got = Oracle.argmax(logits, i * vocab, vocab);
                    assertTrue(
                            expectedLogits[i * vocab + got]
                                    >= expectedLogits[i * vocab + want] - 1e-3f,
                            "position " + i + " picks " + got + " over " + want);
                }
                Oracle.Score logitScore = Oracle.score(expectedLogits, logits, vocab);
                System.out.printf(
                        "worst layer maxRel %.2e, logits %s, worst KL %.2e%n",
                        worst[0], logitScore, worstKl);
                assertTrue(worst[0] < 1e-4, "layer drift " + worst[0]);
                assertTrue(
                        logitScore.maxRel() < 1e-3 && logitScore.minCos() > 0.99999,
                        "logits " + logitScore);
                assertTrue(worstKl < 1e-5, "KL " + worstKl);
            }
        }
    }

    /**
     * The same prompt as ragged prefill chunks - one longer than the window, so the ring wraps
     * inside a chunk and between chunks - then one token at a time: the F16 KV cache is read now,
     * which the one-shot pass never did, so the bound is the F16 cache's, not the arithmetic's.
     */
    @Test
    void chunkedPrefillAndDecodeTrackTheOneShotReference() throws Exception {
        Path dir = Oracle.dir("text-f32");
        int[] tokens = Oracle.ints(dir.resolve("tokens.i32"));
        float[] expected = Oracle.floats(dir.resolve("logits.f32"));
        int[] chunks = {300, 97, 1, 64, 37};
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model = Lfm3.loadModel(TestModels.require(F32), arena);
            int vocab = model.configuration().vocabularySize();
            try (Lfm3.State state = model.newState(tokens.length, 300)) {
                double worstKl = 0;
                int at = 0;
                for (int i = 0; at < tokens.length; i++) {
                    int size = i < chunks.length ? chunks[i] : 1;
                    model.ingest(state, Batch.score(Arrays.copyOfRange(tokens, at, at + size)));
                    for (int row = 0; row < size; row++, at++) {
                        float[] logits =
                                Oracle.rows(
                                        Views.castToSegmentBacked(
                                                model.logits(state, row), "logits"),
                                        1,
                                        vocab);
                        worstKl =
                                Math.max(
                                        worstKl, Oracle.kl(expected, at * vocab, logits, 0, vocab));
                        int want = Oracle.argmax(expected, at * vocab, vocab),
                                got = Oracle.argmax(logits, 0, vocab);
                        assertTrue(
                                expected[at * vocab + got] >= expected[at * vocab + want] - 2e-2f,
                                "position " + at);
                    }
                }
                System.out.printf("chunked + decode: worst KL %.2e%n", worstKl);
                assertTrue(worstKl < 1e-4, "KL " + worstKl);
            }
        }
    }

    @Test
    void theTokenizerAgreesWithLlamaCpp() throws Exception {
        Path dir = Oracle.dir("text-f32");
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model = Lfm3.loadModel(TestModels.require(F32), arena);
            String text = Files.readString(Oracle.ROOT.resolve("prompt-long.txt"));
            int[] expected = Oracle.ints(dir.resolve("tokens.i32"));
            assertArrayEquals(expected, encodeWithSpecials(model.tokenizer(), text));
        }
    }

    /** llama.cpp's parse_special: special-token spellings become their ids, the rest is text. */
    static int[] encodeWithSpecials(Tokenizer tokenizer, String text) {
        IntSequence.Builder out = IntSequence.newBuilder();
        Matcher special = Pattern.compile("<\\|[a-z_]+\\|>|</?think>").matcher(text);
        int at = 0;
        while (special.find()) {
            out.addAll(tokenizer.encode(text.substring(at, special.start())));
            out.add(tokenizer.vocabulary().id(special.group()));
            at = special.end();
        }
        return out.addAll(tokenizer.encode(text.substring(at))).build().toArray();
    }
}
