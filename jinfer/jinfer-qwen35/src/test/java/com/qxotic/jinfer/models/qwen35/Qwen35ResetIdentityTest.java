package com.qxotic.jinfer.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.testkit.ModelFixture;
import com.qxotic.toknroll.IntSequence;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The recycling gate: a {@code reset()} state must generate byte-identically to a fresh one.
 * Qwen3.5 is a sharp case - the gated-delta-net S matrices and the conv history carry values across
 * positions, so a reset that failed to zero either would leak the previous conversation into the
 * next (stale KV rows beyond the cursor are attention-masked and cannot leak). Hot-vs-hot per the
 * numerics convention.
 */
@Tag("integration")
class Qwen35ResetIdentityTest {

    @Test
    void resetStateGeneratesByteIdenticallyToFresh() throws Exception {
        Assumptions.assumeTrue(
                Files.exists(ModelFixture.QWEN35_2B_Q8.path()),
                "model not found: " + ModelFixture.QWEN35_2B_Q8.path());
        Qwen35 model = Qwen35.loadModel(ModelFixture.QWEN35_2B_Q8.path(), Arena.ofAuto());
        var tokenizer = model.loaded().tokenizer();
        IntSequence first = tokenizer.encode("The capital of France is");
        IntSequence second = tokenizer.encode("Once upon a time there was");

        Qwen35.State recycled = new Qwen35.State(model.config(), 1024, 64, Arena.ofAuto());
        // warm the kernels (JIT tier drift flips argmax cold-vs-warm) and DIRTY the state:
        // a full generation leaves real KV rows and real conv residue behind
        for (int i = 0; i < 4; i++) {
            generate(model, recycled, first);
            recycled.reset();
        }
        IntSequence viaReset = generate(model, recycled, second);

        Qwen35.State fresh = new Qwen35.State(model.config(), 1024, 64, Arena.ofAuto());
        IntSequence viaFresh = generate(model, fresh, second);

        assertEquals(
                tokenizer.decode(viaFresh),
                tokenizer.decode(viaReset),
                "reset() must leave no residue: recycled and fresh states must agree");
    }

    private static IntSequence generate(Qwen35 model, Qwen35.State state, IntSequence prompt) {
        return Generator.generate(
                        model,
                        state,
                        prompt,
                        Sampler.ARGMAX,
                        24,
                        0,
                        model.loaded().stopTokens(),
                        token -> true)
                .tokens();
    }
}
