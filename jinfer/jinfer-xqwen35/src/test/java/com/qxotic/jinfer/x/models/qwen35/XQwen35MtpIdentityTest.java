package com.qxotic.jinfer.x.models.qwen35;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.llm.Generator.Constraints;
import com.qxotic.jinfer.x.llm.Sampler;
import com.qxotic.jinfer.x.llm.SpeculativeDecoding.SpeculationResult;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-weight correctness gate for both Qwen3.5 embedded-MTP architectures. */
final class XQwen35MtpIdentityTest {

    @Test
    @Tag("driver")
    void denseMtpCommitsOnlyTargetVerifiedTokens() throws Exception {
        verify("hf.co/unsloth/Qwen3.5-9B-MTP-GGUF:Q4_0", false);
    }

    @Test
    @Tag("driver")
    void moeMtpCommitsOnlyTargetVerifiedTokens() throws Exception {
        verify("hf.co/unsloth/Qwen3.5-35B-A3B-MTP-GGUF:Q8_0", true);
    }

    private static void verify(String ref, boolean moe) throws Exception {
        Path path = TestModels.require(ref);
        Qwen35 model = Qwen35.loadModel(path, Arena.ofAuto());
        assertTrue(model.speculationReady(), "embedded MTP detected");
        assertEquals(moe, model.config().isMoE(), "architecture kind");

        int[] prompt = model.tokenizer().encodeToArray("Count upward: 1, 2, 3,");
        int context = prompt.length + 32;
        try (Qwen35.State state = model.newState(context, Math.max(prompt.length, 8))) {
            model.ingest(state, Batch.prefill(prompt));
            int[] violations = {0};
            SpeculationResult first =
                    model.speculate(
                            state,
                            Sampler.ARGMAX,
                            new Constraints(12, Duration.ZERO, Set.of()),
                            4,
                            null,
                            (token, targetArgmax) -> {
                                if (token != targetArgmax) violations[0]++;
                            });

            assertEquals(0, violations[0], "every emitted token came from the target");
            assertEquals(
                    prompt.length + first.committed().length(),
                    state.position(),
                    "reported committed tokens match target and MTP state");

            state.reset();
            model.ingest(state, Batch.prefill(prompt));
            SpeculationResult replay =
                    model.speculate(
                            state,
                            Sampler.ARGMAX,
                            new Constraints(12, Duration.ZERO, Set.of()),
                            4,
                            null,
                            null);
            assertEquals(first.emitted(), replay.emitted(), "reset restores deterministic MTP");
            assertEquals(first.committed(), replay.committed(), "reset restores cache accounting");
        }
    }
}
