package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Coarse block caching against Qwen3.5 (gated-delta-net hybrid: the S+conv residue is ~2.2MB per
 * linear layer, so cached prompts commit as ONE block per prompt - {@code coarseBlocks}).
 * Model-gated: assume-skips when the file is absent. Run: {@code mvn test
 * -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class Qwen35CoarseCacheIT {

    static final Path MODEL = ModelFixture.QWEN35_2B_Q8.path();

    static final List<org.springframework.ai.chat.messages.Message> PREFIX =
            List.of(new SystemMessage("You are a terse assistant. Answer in one short sentence."));

    @Test
    void cachedPromptWorksByteIdenticallyAndCoarsely() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(48)
                        .build();
        try {
            String question = "What is the capital of France?";
            ChatResponse plain =
                    base.call(new Prompt(List.of(PREFIX.get(0), new UserMessage(question))));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.call(new Prompt(new UserMessage(question)));

            // byte-identity: the restored SSM/KV state produces the same reply as a fresh prefill
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
