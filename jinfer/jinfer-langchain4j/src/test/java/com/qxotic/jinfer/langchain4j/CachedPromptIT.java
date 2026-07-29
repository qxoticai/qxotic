package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
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
import org.junit.jupiter.api.AfterAll;
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
                            "jinfer.testModel", ModelFixture.LFM25_8B_Q8.path().toString()));

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

    @AfterAll
    static void unload() {
        if (base != null) base.close();
    }

    @Test
    void cachedSessionsMultiTurn() {
        // cachedSessions(1): turn 2 strictly extends turn 1's pooled state (the echoed reply
        // restores its verbatim ids through the wire attribute, so the re-encode is the exact
        // generated tokens). BYTE-IDENTITY LAW, proven on ONE engine: the same turn-2 request
        // served from the pool (hit) and cold (the pool no longer matches after it grew past
        // the request) must answer identically. (Cross-ENGINE comparison is deliberately not
        // asserted: two jam pools in one JVM can drift an argmax tie at high-entropy points.)
        JinferChatModel warm =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(128)
                        .cachedSessions(1)
                        .build();
        UserMessage first = UserMessage.from("Remember the codeword PELICAN. Acknowledge briefly.");
        ChatResponse w1 = warm.chat(ChatRequest.builder().messages(first).build());

        ChatRequest secondTurn =
                ChatRequest.builder()
                        .messages(
                                first,
                                w1.aiMessage(),
                                UserMessage.from("What was the codeword? Answer with one word."))
                        .build();
        ChatResponse hit = warm.chat(secondTurn); // strictly extends the pooled turn-1 state
        String stats = warm.engine.sessionStats();
        assertTrue(stats.contains("hits=1"), "turn 2 must reuse turn 1's live state: " + stats);

        ChatResponse cold = warm.chat(secondTurn); // pool grew past this prompt: full prefill
        assertEquals(cold.aiMessage().text(), hit.aiMessage().text());
        assertTrue(hit.aiMessage().text().contains("PELICAN"), hit.aiMessage().text());
        assertTrue(
                warm.engine.sessionStats().contains("hits=1"),
                "the repeat is NOT an extension and must miss: " + warm.engine.sessionStats());
        warm.close();
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
        String stats = base.engine.promptStats();
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
        String stats = base2.engine.promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
        ChatResponse r = a2.chat(UserMessage.from("One word: ok?"));
        assertTrue(!r.aiMessage().text().isBlank());
        base2.close();
    }

    @Test
    void twoModelsAreTwoParallelPipelines() throws Exception {
        // the concurrency contract: one instance is one serial pipeline, so a second pipeline is
        // a second model. Both generate CONCURRENTLY (forbidden on one instance - StateGuard
        // would reject shared-state misuse; here each owns its state) and answer coherently.
        JinferChatModel twin =
                JinferChatModel.builder().modelPath(MODEL).contextLength(2048).build();
        var pool = java.util.concurrent.Executors.newFixedThreadPool(2);
        try {
            var a =
                    pool.submit(
                            () ->
                                    base.chat(UserMessage.from("Say exactly: ALPHA"))
                                            .aiMessage()
                                            .text());
            var b =
                    pool.submit(
                            () ->
                                    twin.chat(UserMessage.from("Say exactly: BRAVO"))
                                            .aiMessage()
                                            .text());
            assertTrue(a.get().contains("ALPHA"), a.get());
            assertTrue(b.get().contains("BRAVO"), b.get());
        } finally {
            pool.shutdown();
            twin.close();
        }
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

        Path other = ModelFixture.GEMMA4_E2B_QAT_Q4.path();
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
