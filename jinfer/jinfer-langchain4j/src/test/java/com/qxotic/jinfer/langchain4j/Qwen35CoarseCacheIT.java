package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link AbstractCoarseCacheIT} on Qwen3.5 (gated-delta-net hybrid: the S-matrix residue is ~2.1MB
 * per linear layer), plus the Qwen-only artifact round trip and the long chunked-prefix seam.
 * Model-gated: assume-skips when the file is absent.
 */
@Tag("integration")
class Qwen35CoarseCacheIT extends AbstractCoarseCacheIT {

    private static final String REF = "hf.co/unsloth/Qwen3.5-2B-GGUF/Qwen3.5-2B-Q8_0.gguf";

    @Override
    Path modelPath() {
        return TestModels.require(REF);
    }

    /**
     * Byte-identity is a law only under DETERMINISTIC decode: the builder otherwise takes the
     * model's recommended temperature, and two sampled runs never match.
     */
    private static JinferChatModel.Builder deterministic() {
        return JinferChatModel.builder()
                .modelPath(TestModels.require(REF))
                .contextLength(4096)
                .maxOutputTokens(32)
                .temperature(0.0)
                .seed(7L);
    }

    /** The artifact round trip: export the defined prompt, remount it, still byte-identical. */
    @Test
    void frozenArtifactRoundTrips() throws Exception {
        Path artifact = Files.createTempDirectory("qwen35-coarse").resolve("prefix.jkvf");
        String question = "Name one primary color.";
        String direct;
        JinferChatModel base = deterministic().build();
        try {
            JinferChatModel view = base.withCachedPrompt(PREFIX, List.of());
            direct = view.chat(UserMessage.from(question)).aiMessage().text();
            base.saveCachedPrompts(artifact);
        } finally {
            base.close();
        }

        JinferChatModel mounted = deterministic().promptCache(artifact).build();
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
                        AiMessage.from("Consult knowledge article 1007 before answering."),
                        UserMessage.from("And for case class 42? One sentence."));
        JinferChatModel base = deterministic().build();
        try {
            base.chat(UserMessage.from("warmup"));
            String question = "Which article covers case class 13? One sentence.";
            String plain =
                    base.chat(
                                    ChatRequest.builder()
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
        List<ChatMessage> all = new ArrayList<>(prefix);
        all.add(last);
        return all;
    }
}
