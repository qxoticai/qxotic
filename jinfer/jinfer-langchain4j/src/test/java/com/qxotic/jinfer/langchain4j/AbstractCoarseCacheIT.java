package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Coarse block caching, shared across the hybrid models whose codec overrides {@code coarseBlocks}
 * (the recurrent-state residue is MBs per block, so cached prompts commit as ONE block per prompt).
 * Subclasses supply the model; each is model-gated and assume-skips when the file is absent. The
 * BYTE-level gate (restored state vs live state via Harness.statesEqual) lives in the model
 * modules' CacheRun drivers; this asserts reply-text identity under greedy decode.
 */
abstract class AbstractCoarseCacheIT {

    static final List<ChatMessage> PREFIX =
            List.of(SystemMessage.from("You are a terse assistant. Answer in one short sentence."));

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
                        .maxOutputTokens(48)
                        .build();
        try {
            base.chat(UserMessage.from("warmup")); // JIT-warm the kernels before the baseline
            String question = "What is the capital of France?";
            ChatResponse plain = base.chat(PREFIX.get(0), UserMessage.from(question));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.chat(UserMessage.from(question));

            // the restored recurrent/KV state answers exactly like a fresh prefill (greedy text)
            assertEquals(plain.aiMessage().text(), cached.aiMessage().text());

            // the view request restored the prefix from the tree (no cache-read usage field in
            // langchain4j's TokenUsage - the engine stats are the observable)
            String stats = base.engine.promptStats();
            assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);

            // coarse: exactly ONE block per defined prompt (one residue, not one per turn),
            // and a served turn must never add another
            assertTrue(stats.startsWith("blocks=1 "), stats);
        } finally {
            base.close();
        }
    }
}
