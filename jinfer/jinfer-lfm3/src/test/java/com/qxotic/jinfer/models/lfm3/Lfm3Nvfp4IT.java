package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * An NVFP4 file end to end, prefill and decode: its feed-forward weights carry a {@code .scale}
 * scalar each (about 2e-4), and a reply without them, or with the codes' half missing, is noise.
 */
@Tag("integration")
final class Lfm3Nvfp4IT {

    @Test
    void anNvfp4ModelKeepsCounting() throws Exception {
        Lfm3 model =
                Lfm3.loadModel(
                        TestModels.require("hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-NVFP4.gguf"),
                        Arena.ofAuto());
        int vocab = model.configuration().vocabularySize();
        int[] prompt =
                model.tokenizer()
                        .encode("1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,")
                        .toArray();
        int[] reply = new int[8];
        float[] logits = new float[vocab];
        try (Lfm3.State state = model.newState(64, 64)) {
            model.ingest(state, Batch.prefill(prompt));
            for (int step = 0; step < reply.length; step++) {
                Views.copyToArray(
                        Views.castToSegmentBacked(model.logits(state, 0), "logits"),
                        0,
                        logits,
                        0,
                        vocab,
                        "logits");
                reply[step] = Oracle.argmax(logits, 0, vocab);
                model.ingest(state, Batch.step(reply[step]));
            }
        }
        String text = model.tokenizer().decode(reply);
        assertTrue(text.startsWith(" 19, 20, 21"), "the reply was '" + text + "'");
    }
}
