package com.qxotic.jinfer.x.models.gptoss;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.jinfer.x.kernels.Ops;
import com.qxotic.jota.memory.MemoryView;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class XGptOssTest {
    private static final int TOKENS = 4;
    private static final float LOGIT_TOLERANCE = 1e-2f;

    @Test
    void greedyParity() throws Exception {
        assertGreedyParity(TestModels.require("hf.co/unsloth/gpt-oss-20b-GGUF:Q8_0"));
    }

    private static void assertGreedyParity(Path model) throws Exception {
        try (FileChannel channel = FileChannel.open(model)) {
            GGUF gguf = ModelLoader.readGguf(channel, model.toString());
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            int[] prompt = tokenizer.encodeToArray("The capital of France is");
            var old =
                    com.qxotic.jinfer.models.gptoss.GptOss.loadModel(
                            channel, gguf, Arena.ofAuto(), tokenizer);
            var x = GptOss.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            int context = Math.min(old.config().contextLength(), prompt.length + TOKENS + 8);
            int batch = Math.max(16, prompt.length);
            int vocab = old.config().vocabularySize();

            try (var oldState = old.newState(context, batch);
                    var xState = x.newState(context, batch)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(prompt));
                x.ingest(xState, Batch.prefill(prompt));
                FloatTensor oldLogits = old.logits(oldState);
                MemoryView<MemorySegment> xLogits =
                        Views.castToSegmentBacked(x.logits(xState), "logits");
                assertLogits(oldLogits, xLogits, vocab, "prefill");
                for (int i = 0; i < TOKENS; i++) {
                    int oldToken = oldLogits.argmax();
                    int xToken = Ops.argmax(xLogits, 0, vocab);
                    assertEquals(oldToken, xToken, "greedy token divergence at step " + i);
                    old.ingest(oldState, com.qxotic.jinfer.Batch.step(oldToken));
                    x.ingest(xState, Batch.step(xToken));
                    oldLogits = old.logits(oldState);
                    xLogits = Views.castToSegmentBacked(x.logits(xState), "logits");
                    assertLogits(oldLogits, xLogits, vocab, "decode step " + i);
                }
            }
            try (var oldState = old.newState(context, batch);
                    var xState = x.newState(context, batch)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.score(prompt));
                x.ingest(xState, Batch.score(prompt));
                for (int i = 0; i < prompt.length; i++)
                    assertLogits(
                            old.logits(oldState, i),
                            Views.castToSegmentBacked(x.logits(xState, i), "logits"),
                            vocab,
                            "score row " + i);
            }
        }
    }

    private static void assertLogits(
            FloatTensor old, MemoryView<MemorySegment> x, int n, String stage) {
        float max = 0f;
        for (int i = 0; i < n; i++)
            max = Math.max(max, Math.abs(old.getFloat(i) - Views.getFloat(x, i, "logits")));
        assertTrue(max < LOGIT_TOLERANCE, stage + " logits diverged: " + max);
    }
}
