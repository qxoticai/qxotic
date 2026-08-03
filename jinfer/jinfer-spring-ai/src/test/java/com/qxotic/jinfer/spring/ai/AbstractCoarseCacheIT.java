package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Coarse block caching, shared across the hybrid models whose codec overrides {@code coarseBlocks}
 * (the recurrent-state residue is MBs per block, so cached prompts commit as ONE block per prompt).
 * Subclasses supply the model; each is model-gated and assume-skips when the file is absent.
 */
abstract class AbstractCoarseCacheIT {

    static final List<Message> PREFIX =
            List.of(new SystemMessage("You are a terse assistant. Answer in one short sentence."));

    /** The coarse-codec model under test. */
    abstract Path modelPath();

    @Test
    @Tag("integration")
    void cachedPromptWorksByteIdenticallyAndCoarsely() {
        Path model = modelPath();
        Assumptions.assumeTrue(Files.exists(model), "model not found: " + model);
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(model)
                        .contextLength(4096)
                        .maxTokens(48)
                        .build();
        try {
            // JIT-warm the kernels before the baseline (cold passes differ by ~1 LSB)
            base.call(new Prompt(new UserMessage("warmup")));
            String question = "What is the capital of France?";
            ChatResponse plain =
                    base.call(new Prompt(List.of(PREFIX.get(0), new UserMessage(question))));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.call(new Prompt(new UserMessage(question)));

            // byte-identity: the restored recurrent/KV state answers like a fresh prefill
            assertEquals(
                    plain.getResult().getOutput().getText(),
                    cached.getResult().getOutput().getText());

            // the cache read is reported, and it covered the prefix
            Long cacheRead = cached.getMetadata().getUsage().getCacheReadInputTokens();
            assertNotNull(cacheRead, "a view request must report the restored prefix");
            assertTrue(cacheRead > 0, "restored nothing: " + cacheRead);

            // coarse: exactly ONE block per defined prompt (one residue, not one per turn)
            String stats = base.engine.promptStats();
            assertTrue(stats.startsWith("blocks=1 "), stats);
        } finally {
            base.close();
        }
    }
}
