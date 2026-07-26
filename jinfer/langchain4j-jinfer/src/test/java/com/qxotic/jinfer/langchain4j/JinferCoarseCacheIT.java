package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
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
 * Coarse block caching against Nemotron-H (Mamba2 hybrid: the SSM residue is ~90MB per block at 30B
 * dims, so cached prompts commit as ONE block per prompt - {@code coarseBlocks}). Model-gated (30B
 * - slow to load): assume-skips when the file is absent. Run: {@code mvn test
 * -Dsurefire.excludedGroups= -Dgroups=integration -pl langchain4j-jinfer}
 */
@Tag("integration")
class JinferCoarseCacheIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModelCoarse",
                            ModelFixture.NEMOTRON_30B_Q8.path().toString()));

    static final List<ChatMessage> PREFIX =
            List.of(SystemMessage.from("You are a terse assistant. Answer in one short sentence."));

    @Test
    void cachedPromptWorksByteIdenticallyAndCoarsely() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(48)
                        .build();
        try {
            String question = "What is the capital of France?";
            ChatResponse plain = base.chat(PREFIX.get(0), UserMessage.from(question));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.chat(UserMessage.from(question));

            // byte-identity: the restored SSM/KV state produces the same reply as a fresh prefill
            assertEquals(plain.aiMessage().text(), cached.aiMessage().text());

            // the view request restored the prefix from the tree (no cache-read usage field in
            // langchain4j's TokenUsage - the engine stats are the observable)
            String stats = base.engine.promptStats();
            assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);

            // coarse: exactly ONE block per defined prompt (one residue, not one per turn)
            assertTrue(stats.startsWith("blocks=1 "), stats);
        } finally {
            base.close();
        }
    }
}
