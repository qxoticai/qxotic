package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.kernels.Convert;
import com.qxotic.jinfer.kernels.Ops;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Media rows are committed rows like any other: the MTP block's KV prefix and the target hidden
 * behind {@code logits} must cover them. An embedding batch carrying exactly the token embeddings
 * is the reference-free probe: it has to produce the token batch's logits and the token batch's
 * draft.
 */
final class Qwen35MtpMediaRowsTest {

    @Test
    @Tag("driver")
    void embeddingRowsFeedTheDraftHeadLikeTokens() throws Exception {
        Path path = TestModels.require("hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_0");
        Path mmproj = TestModels.require("hf.co/unsloth/Qwen3.5-9B-MTP-GGUF/mmproj-F16.gguf");
        try (Arena arena = Arena.ofConfined()) {
            Qwen35 model = Qwen35.loadModel(path, arena).withMedia(mmproj, arena);
            int dim = model.configuration().embeddingLength();
            int vocab = model.configuration().vocabularySize();
            int[] prompt = model.tokenizer().encodeToArray("Count upward: 1, 2, 3,");
            int split = prompt.length / 2;
            int[] tail = Arrays.copyOfRange(prompt, split, prompt.length);
            MemoryView<MemorySegment> rows =
                    Views.allocateF32(MemoryAllocators.ofArena(arena), tail.length, dim);
            Convert.gatherToF32(
                    model.weights().tokenEmbedding(), tail, 0, tail.length, rows, 0, dim);

            try (Qwen35.State state = model.newState(prompt.length + 8, prompt.length)) {
                model.ingest(state, Batch.prefill(prompt));
                int[] viaTokens = new int[4];
                viaTokens[0] =
                        Ops.argmax(
                                Views.castToSegmentBacked(model.logits(state, 0), "logits"),
                                0,
                                vocab);
                model.draft(state, 3, viaTokens);

                state.reset();
                model.ingest(state, Batch.prefill(Arrays.copyOf(prompt, split)));
                model.ingest(state, Batch.embeddings(rows, tail.length));
                assertEquals(prompt.length, state.position());
                int[] viaRows = new int[4];
                viaRows[0] =
                        Ops.argmax(
                                Views.castToSegmentBacked(model.logits(state, 0), "logits"),
                                0,
                                vocab);
                model.draft(state, 3, viaRows);

                assertArrayEquals(
                        viaTokens, viaRows, "target argmax and MTP draft after media rows");
            }
        }
    }
}
