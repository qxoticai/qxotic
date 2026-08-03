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
 * linear layer, so cached prompts commit as ONE block per prompt - {@code coarseBlocks}). Asserts
 * reply-text identity under greedy decode; the BYTE-level gate (restored state vs live state via
 * Harness.statesEqual) lives in Qwen35CacheRun on the model module. Model-gated: assume-skips when
 * the file is absent.
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
            base.chat(UserMessage.from("warmup")); // JIT-warm the kernels before the baseline
            String question = "What is the capital of France?";
            ChatResponse plain = base.chat(PREFIX.get(0), UserMessage.from(question));

            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            ChatResponse cached = view.chat(UserMessage.from(question));

            // the restored S/conv/KV state answers exactly like a fresh prefill (greedy text)
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
            // and the artifact was USED - a silent recompute-from-scratch would also match
            String stats = mounted.engine.promptStats();
            assertTrue(
                    stats.contains("hits=") && !stats.contains("hits=0 "),
                    "the mounted artifact must serve the restore: " + stats);
        } finally {
            mounted.close();
        }
    }

    /**
     * A LONG multi-turn prefix (well past the 512-token ingest chunk): define forwards it as ONE
     * chunk while a fresh serve forwards per-turn chunks, so this pins byte-identity across the
     * chunk-shape seam the short test cannot see.
     */
    @Test
    void longChunkedPrefixStaysIdentical() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        StringBuilder manual = new StringBuilder("You are the AcmeCloud support agent.");
        for (int i = 1; i <= 90; i++) {
            manual.append(" Rule ")
                    .append(i)
                    .append(": for case class ")
                    .append(i)
                    .append(" consult knowledge article ")
                    .append(1000 + i)
                    .append(" before answering.");
        }
        List<ChatMessage> prefix =
                List.of(
                        SystemMessage.from(manual.toString()),
                        UserMessage.from("What do I do for case class 7? One sentence."),
                        dev.langchain4j.data.message.AiMessage.from(
                                "Consult knowledge article 1007 before answering."),
                        UserMessage.from("And for case class 42? One sentence."));
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(32)
                        .build();
        try {
            base.chat(UserMessage.from("warmup"));
            String question = "Which article covers case class 13? One sentence.";
            String plain =
                    base.chat(
                                    dev.langchain4j.model.chat.request.ChatRequest.builder()
                                            .messages(append(prefix, UserMessage.from(question)))
                                            .build())
                            .aiMessage()
                            .text();
            JinferChatModel view = base.withCachedPrompt(prefix, List.of());
            String cached = view.chat(UserMessage.from(question)).aiMessage().text();
            assertEquals(plain, cached, "chunk-shape seam: one-chunk define vs per-turn serve");
            assertTrue(
                    base.engine.promptStats().startsWith("blocks=1 "), base.engine.promptStats());
        } finally {
            base.close();
        }
    }

    private static List<ChatMessage> append(List<ChatMessage> prefix, ChatMessage last) {
        java.util.List<ChatMessage> all = new java.util.ArrayList<>(prefix);
        all.add(last);
        return all;
    }
}
