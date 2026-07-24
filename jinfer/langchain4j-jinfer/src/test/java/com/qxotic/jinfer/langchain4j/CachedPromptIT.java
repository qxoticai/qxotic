package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Cached prompts end-to-end (LFM2.5-8B): the byte-identity law (a cached prompt may never change
 * output), zero-prefill re-declaration through a saved artifact, shared-prefix dedup, and the
 * loud-failure contracts. Model-gated: assume-skips without the GGUF.
 */
@Tag("integration")
class CachedPromptIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel",
                            "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf"));

    static final List<ChatMessage> SUPPORT =
            List.of(
                    SystemMessage.from(
                            "You are a terse support assistant for AcmeCloud. Answer in one"
                                + " sentence. The reset portal is https://acme.example/reset."));

    static JinferChatModel base;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(128)
                        .build();
    }

    @Test
    void byteIdentityWithUncached() {
        String question = "Where do I reset my password?";
        // uncached: prefix inlined into the request on the BASE model (which never uses the tree)
        ChatResponse plain = base.chat(SUPPORT.get(0), UserMessage.from(question));
        // cached: same conversation through a view
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        ChatResponse cached = support.chat(UserMessage.from(question));

        assertEquals(plain.aiMessage().text(), cached.aiMessage().text());
        assertTrue(
                cached.aiMessage().text().contains("acme.example/reset"),
                cached.aiMessage().text());
    }

    @Test
    void treeIsConsultedAndBaseStaysCold() {
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        support.chat(UserMessage.from("Hello?"));
        String stats = base.engine().promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
    }

    @Test
    void sharedPrefixDedupAndArtifactRoundTrip() throws Exception {
        String common = "You are an assistant for AcmeCloud. Be brief. ";
        JinferChatModel a =
                base.withCachedPrompt(
                        List.of(SystemMessage.from(common + "You handle SUPPORT tickets.")),
                        List.of());
        JinferChatModel b =
                base.withCachedPrompt(
                        List.of(SystemMessage.from(common + "You handle SALES questions.")),
                        List.of());

        Path artifact = Files.createTempDirectory("cached-prompts").resolve("personas.jkv");
        base.saveCachedPrompts(artifact);
        assertTrue(Files.size(artifact) > 0);

        // fresh engine, mounted artifact: re-declaration matches frozen blocks (hits, no misses
        // beyond the unavoidable tail), and answers still flow
        JinferChatModel base2 =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(64)
                        .loadCachedPrompts(artifact)
                        .build();
        JinferChatModel a2 =
                base2.withCachedPrompt(
                        List.of(SystemMessage.from(common + "You handle SUPPORT tickets.")),
                        List.of());
        String stats = base2.engine().promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
        ChatResponse r = a2.chat(UserMessage.from("One word: ok?"));
        assertTrue(!r.aiMessage().text().isBlank());
    }

    @Test
    void viewRejectsRequestTools() {
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        ToolSpecification t =
                ToolSpecification.builder()
                        .name("noop")
                        .parameters(JsonObjectSchema.builder().build())
                        .build();
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        support.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .toolSpecifications(t)
                                        .build()));
    }

    @Test
    void wrongModelArtifactFailsAtBuild() throws Exception {
        Path artifact = Files.createTempDirectory("cached-prompts").resolve("lfm2.jkv");
        base.withCachedPrompt(SUPPORT, List.of());
        base.saveCachedPrompts(artifact);

        Path other =
                Path.of(
                        "/home/mukel/Desktop/playground/models/unsloth/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
        Assumptions.assumeTrue(Files.exists(other), "second model not found");
        assertThrows(
                Exception.class,
                () ->
                        JinferChatModel.builder()
                                .modelPath(other)
                                .contextLength(2048)
                                .loadCachedPrompts(artifact)
                                .build());
    }
}
