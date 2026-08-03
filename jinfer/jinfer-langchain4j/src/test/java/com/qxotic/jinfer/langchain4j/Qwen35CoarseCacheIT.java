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
 * Coarse block caching against Qwen3.5 (gated-delta-net hybrid: the S-matrix residue is ~2.1MB per
 * linear layer, so cached prompts commit as ONE block per prompt - {@code coarseBlocks}). The
 * byte-identity assertion is the codec's whole law: the restored S/conv/KV state must produce the
 * same reply as a fresh prefill, bit for bit. Model-gated: assume-skips when the file is absent.
 */
@Tag("integration")
class Qwen35CoarseCacheIT {

    static final Path MODEL = ModelFixture.QWEN35_2B_Q8.path();

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

            // byte-identity: the restored S/conv/KV state answers exactly like a fresh prefill
            assertEquals(plain.aiMessage().text(), cached.aiMessage().text());

            String stats = base.engine.promptStats();
            assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);

            // coarse: exactly ONE block per defined prompt (one residue, not one per turn),
            // and a served turn must never add another
            assertTrue(stats.startsWith("blocks=1 "), stats);
        } finally {
            base.close();
        }
    }

    /** The artifact round trip: export the defined prompt, remount it, still byte-identical. */
    @Test
    void frozenArtifactRoundTrips() throws Exception {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        Path artifact = Files.createTempDirectory("qwen35-coarse").resolve("prefix.jkvf");
        String question = "Name one primary color.";
        String direct;
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(32)
                        .build();
        try {
            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            direct = view.chat(UserMessage.from(question)).aiMessage().text();
            base.saveCachedPrompts(artifact);
        } finally {
            base.close();
        }

        JinferChatModel mounted =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(32)
                        .loadCachedPrompts(artifact)
                        .build();
        try {
            JinferChatModel view = mounted.withCachedPrompt(PREFIX, List.of());
            String restored = view.chat(UserMessage.from(question)).aiMessage().text();
            assertEquals(direct, restored, "the S/conv residue must survive the disk round trip");
        } finally {
            mounted.close();
        }
    }
}
