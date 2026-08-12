package com.qxotic.jinfer.x.models.nemotronh;

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
import java.util.Arrays;
import org.junit.jupiter.api.Test;

class XNemotronHTest {
    private static final String REF = "hf.co/bartowski/nvidia_Nemotron-Cascade-2-30B-A3B-GGUF:Q8_0";
    private static final float TOLERANCE = 1e-2f;

    @Test
    void greedyAllOutputAndResetParity() throws Exception {
        Path path = TestModels.require(REF);
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, path.toString());
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            int[] prompt = tokenizer.encodeToArray("The capital of France is");
            var old =
                    com.qxotic.jinfer.models.nemotronh.NemotronH.loadModel(
                            channel, gguf, Arena.ofAuto(), tokenizer);
            var x = NemotronH.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            int context = prompt.length + 8, batch = Math.max(16, prompt.length);
            try (var oldState = old.newState(context, batch);
                    var xState = x.newState(context, batch)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(prompt));
                x.ingest(xState, Batch.prefill(prompt));
                FloatTensor oldLogits = old.logits(oldState);
                MemoryView<MemorySegment> xLogits = logits(x, xState, 0);
                assertLogits(oldLogits, xLogits, "prefill");
                float[] first = Views.toFloatArray(xLogits, "logits");
                for (int step = 0; step < 2; step++) {
                    int oldToken = oldLogits.argmax(),
                            xToken = Ops.argmax(xLogits, 0, x.config().vocabularySize());
                    assertEquals(oldToken, xToken, "greedy token at step " + step);
                    old.ingest(oldState, com.qxotic.jinfer.Batch.step(oldToken));
                    x.ingest(xState, Batch.step(xToken));
                    oldLogits = old.logits(oldState);
                    xLogits = logits(x, xState, 0);
                    assertLogits(oldLogits, xLogits, "decode step " + step);
                }
                oldState.reset();
                xState.reset();
                int[] score = Arrays.copyOf(prompt, Math.min(3, prompt.length));
                old.ingest(oldState, com.qxotic.jinfer.Batch.score(score));
                x.ingest(xState, Batch.score(score));
                for (int output = 0; output < score.length; output++)
                    assertLogits(
                            old.logits(oldState, output),
                            logits(x, xState, output),
                            "all output " + output);
                xState.reset();
                x.ingest(xState, Batch.prefill(prompt));
                float[] reset = Views.toFloatArray(logits(x, xState, 0), "logits");
                float max = 0f;
                for (int i = 0; i < first.length; i++)
                    max = Math.max(max, Math.abs(first[i] - reset[i]));
                assertTrue(max < TOLERANCE, "reset prefill logits diverged: " + max);
            }
        }
    }

    private static MemoryView<MemorySegment> logits(
            NemotronH model, NemotronH.State state, int output) {
        return Views.castToSegmentBacked(model.logits(state, output), "logits");
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
