package com.qxotic.jinfer.x.models.qwen35;

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

class XQwen35Test {
    private static final String REF = "hf.co/unsloth/Qwen3.5-2B-GGUF:Q8_0";
    private static final float TOLERANCE = 1e-2f;

    @Test
    void greedyParityAndResetIdentity() throws Exception {
        Path path = TestModels.require(REF);
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            int[] prompt = tokenizer.encodeToArray("The capital of France is");
            var old =
                    com.qxotic.jinfer.models.qwen35.Qwen35.loadModel(
                            channel, gguf, Arena.ofAuto(), tokenizer);
            var x = Qwen35.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            int context = prompt.length + 16, batch = Math.max(16, prompt.length);
            try (var oldState = old.newState(context, batch);
                    var xState = x.newState(context, batch)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(prompt));
                x.ingest(xState, Batch.prefill(prompt));
                FloatTensor oldLogits = old.logits(oldState);
                MemoryView<MemorySegment> xLogits = logits(x, xState);
                assertLogits(oldLogits, xLogits, "prefill");
                float[] first = Views.toFloatArray(xLogits, "logits");
                for (int step = 0; step < 4; step++) {
                    int oldToken = oldLogits.argmax();
                    int xToken = Ops.argmax(xLogits, 0, x.config().vocabularySize());
                    assertEquals(oldToken, xToken, "greedy token at step " + step);
                    old.ingest(oldState, com.qxotic.jinfer.Batch.step(oldToken));
                    x.ingest(xState, Batch.step(xToken));
                    oldLogits = old.logits(oldState);
                    xLogits = logits(x, xState);
                    assertLogits(oldLogits, xLogits, "decode step " + step);
                }
                xState.reset();
                x.ingest(xState, Batch.prefill(prompt));
                float[] reset = Views.toFloatArray(logits(x, xState), "logits");
                assertEquals(first.length, reset.length);
                float max = 0f;
                for (int i = 0; i < first.length; i++)
                    max = Math.max(max, Math.abs(first[i] - reset[i]));
                assertTrue(max < TOLERANCE, "reset prefill logits diverged: " + max);

                xState.reset();
                x.ingest(xState, Batch.score(prompt));
                assertEquals(prompt.length, xState.outputCount(), "ALL retains every target row");
                float[] scored = Views.toFloatArray(logits(x, xState), "logits");
                max = 0f;
                for (int i = 0; i < first.length; i++)
                    max = Math.max(max, Math.abs(first[i] - scored[i]));
                assertTrue(max < TOLERANCE, "ALL's last row diverged from LAST: " + max);
            }
        }
    }

    private static MemoryView<MemorySegment> logits(Qwen35 model, Qwen35.State state) {
        return Views.castToSegmentBacked(model.logits(state), "logits");
    }

    private static void assertLogits(
            FloatTensor expected, MemoryView<MemorySegment> actual, String stage) {
        float max = 0f;
        for (int i = 0; i < expected.size(); i++)
            max =
                    Math.max(
                            max,
                            Math.abs(expected.getFloat(i) - Views.getFloat(actual, i, "logits")));
        assertTrue(max < TOLERANCE, stage + " logits diverged: " + max);
    }
}
