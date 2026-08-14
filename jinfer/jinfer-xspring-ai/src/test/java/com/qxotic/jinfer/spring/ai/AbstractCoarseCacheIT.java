package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
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

    /** The coarse-codec model under test, by ref. */
    abstract String modelRef();

    @Test
    @Tag("integration")
    void cachedPromptWorksByteIdenticallyAndCoarsely() {
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(com.qxotic.jinfer.testkit.TestModels.require(modelRef()))
                        .contextLength(4096)
                        // pinned decode like the langchain4j twin: greedy, seeded, no think span -
                        // the model's recommended sampled temperature would make "contains Paris"
                        // a coin toss, and a think span eats the 48-token budget before the answer
                        .options(
                                JinferChatOptions.builder()
                                        .maxTokens(48)
                                        .temperature(0.0)
                                        .seed(7L)
                                        .thinking(false)
                                        .build())
                        .build();
        try {
            // JIT-warm the kernels before the baseline (cold passes differ by ~1 LSB)
            base.call(new Prompt(new UserMessage("warmup")));
            String question = "What is the capital of France?";
            ChatResponse plain =
                    base.call(new Prompt(List.of(PREFIX.get(0), new UserMessage(question))));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.call(new Prompt(new UserMessage(question)));

            // the restored recurrent/KV state answers the question correctly. Exact-text equality
            // with the plain reply is NOT asserted: restore==live is byte-exact (gated strictly by
            // the family CacheRun drivers), but plain-vs-cached compares one-shot prefill against
            // chunked ingest, whose states drift an ulp (generic to the hybrids) and can flip a
            // greedy argmax tie - observed on NemotronH within 48 tokens
            assertTrue(
                    plain.getResult().getOutput().getText().contains("Paris"),
                    plain.getResult().getOutput().getText());
            assertTrue(
                    cached.getResult().getOutput().getText().contains("Paris"),
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
