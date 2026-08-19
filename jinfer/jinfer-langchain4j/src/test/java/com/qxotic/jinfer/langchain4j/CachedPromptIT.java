package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.Tool;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.agent.tool.ToolSpecifications;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.service.AiServices;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
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

    private static final String REF = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";

    static final List<ChatMessage> SUPPORT =
            List.of(
                    SystemMessage.from(
                            "You are a terse support assistant for AcmeCloud. Answer in one"
                                + " sentence. The reset portal is https://acme.example/reset."));

    static JinferChatModel base;

    @BeforeAll
    static void load() {
        base =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        .maxOutputTokens(128)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (base != null) base.close();
    }

    @Test
    void retainedSessionsResumeMultiTurn() {
        // retainSessions(1): turn 2 strictly extends turn 1's retained state (the echoed reply
        // restores its verbatim ids through the wire attribute, so the re-encode is the exact
        // generated tokens). BYTE-IDENTITY LAW, proven on ONE engine: the same turn-2 request
        // served from the pool (hit) and cold (the pool no longer matches after it grew past
        // the request) must answer identically. (Cross-ENGINE comparison is deliberately not
        // asserted: two jam pools in one JVM can drift an argmax tie at high-entropy points.)
        // pinned seed: hit-vs-cold is a byte-identity comparison, and an unlucky random seed can
        // spend the whole budget on a think span (null text)
        JinferChatModel warm =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        .maxOutputTokens(128)
                        .retainSessions(1)
                        .seed(7L)
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
        // OWN engine + pinned seed: the law is view-vs-inline on the SAME tree state, and block
        // KV is bit-exact only against the define that produced it - other tests' defines share
        // head blocks computed in different batch shapes, which can drift an ulp and flip an
        // argmax near-tie. Self-contained, the comparison tests the law, not test order.
        JinferChatModel fresh =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        .maxOutputTokens(128)
                        .seed(7L)
                        .build();
        try {
            String question = "Where do I reset my password?";
            // uncached: prefix inlined into the request on the BASE model (never uses the tree)
            ChatResponse plain = fresh.chat(SUPPORT.get(0), UserMessage.from(question));
            // cached: same conversation through a view
            JinferChatModel support = fresh.withCachedPrompt(SUPPORT, List.of());
            ChatResponse cached = support.chat(UserMessage.from(question));

            assertEquals(plain.aiMessage().text(), cached.aiMessage().text());
            assertTrue(
                    cached.aiMessage().text().contains("acme.example/reset"),
                    cached.aiMessage().text());
        } finally {
            fresh.close();
        }
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
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        .maxOutputTokens(64)
                        .promptCache(artifact)
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
    void forkIsAParallelPipelineSharingTheWeights() throws Exception {
        // ONE load in the USER's arena; fork() mints a second pipeline for the price of a
        // context. The arena's block structure IS the ownership story: weights die at the brace.
        try (Arena arena = Arenas.newCrossThread()) {
            var loaded = Models.load(TestModels.require(REF), arena);
            // contextLength is a load-time setting on the LoadedModel path - the builder
            // refuses it here by design; a generous budget keeps the echo assertion off luck
            JinferChatModel a =
                    JinferChatModel.builder()
                            .model(loaded)
                            .maxOutputTokens(128)
                            .thinking(false)
                            .seed(1L)
                            .build();
            JinferChatModel b = a.fork();
            try {
                var pool = Executors.newFixedThreadPool(2);
                try {
                    var fa =
                            pool.submit(
                                    () ->
                                            a.chat(UserMessage.from("Say exactly: ALPHA"))
                                                    .aiMessage()
                                                    .text());
                    var fb =
                            pool.submit(
                                    () ->
                                            b.chat(UserMessage.from("Say exactly: BRAVO"))
                                                    .aiMessage()
                                                    .text());
                    assertTrue(fa.get().contains("ALPHA"), fa.get());
                    assertTrue(fb.get().contains("BRAVO"), fb.get());
                } finally {
                    pool.shutdown();
                }
                // a view's fork re-defines the prefix on its own tree and serves from it
                JinferChatModel view = a.withCachedPrompt(SUPPORT, List.of());
                JinferChatModel forkedView = view.fork();
                try {
                    String answer =
                            forkedView
                                    .chat(UserMessage.from("Where do I reset my password?"))
                                    .aiMessage()
                                    .text();
                    assertTrue(answer != null && !answer.isBlank());
                } finally {
                    forkedView.close(); // closes the fork's engine; a and its views live on
                }
            } finally {
                a.close();
                b.close();
            }
        }
    }

    @Test
    void useAfterTheOwnerFreesTheWeightsFailsFast() throws Exception {
        // the 350M keeps this cheap; the canary at the forward's entry turns what used to be a
        // SIGSEGV into a teaching ISE - for the SEQUENTIAL mistake only
        Arena arena = Arenas.newCrossThread();
        JinferChatModel borrowed;
        try {
            var loaded =
                    Models.load(
                            TestModels.require(
                                    "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"),
                            arena);
            borrowed = JinferChatModel.builder().model(loaded).maxOutputTokens(16).build();
        } catch (Throwable t) {
            arena.close();
            throw t;
        }
        try {
            arena.close(); // the owner frees the weights under the pipeline - out of order
            IllegalStateException e =
                    assertThrows(
                            IllegalStateException.class,
                            () -> borrowed.chat(UserMessage.from("hi")));
            assertTrue(e.getMessage().contains("freed"), e.getMessage());
        } finally {
            borrowed.close();
        }
    }

    @Test
    void forkOfAnOwningModelRefusesWithTheRecipe() {
        IllegalStateException e = assertThrows(IllegalStateException.class, base::fork);
        assertTrue(e.getMessage().contains("Models.load"), e.getMessage());
        assertTrue(e.getMessage().contains("model(loaded)"), e.getMessage());
    }

    @Test
    void twoModelsAreTwoParallelPipelines() throws Exception {
        // the concurrency contract: one instance is one serial pipeline, so a second pipeline is
        // a second model. Both generate CONCURRENTLY (forbidden on one instance - StateGuard
        // would reject shared-state misuse; here each owns its state) and answer coherently.
        JinferChatModel twin =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(2048)
                        .build();
        var pool = Executors.newFixedThreadPool(2);
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

    static final ToolSpecification RESET =
            ToolSpecification.builder()
                    .name("open_reset_portal")
                    .description("Open the password reset portal for a customer email")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("email", "The customer's email address")
                                    .required("email")
                                    .build())
                    .build();

    @Test
    void identicalRequestToolsServeFromTheCache() {
        // a view's tools are its DEFAULT tool set; a request re-stating the same set (what
        // AiServices does every call) is a cache hit, and the usage accounting proves it
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of(RESET));
        ChatResponse r =
                support.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Say OK."))
                                .toolSpecifications(RESET)
                                .build());
        JinferTokenUsage usage = (JinferTokenUsage) r.tokenUsage();
        assertTrue(
                usage.cachedInputTokens() > 0,
                "the welded prefix must be restored, not re-prefilled: " + usage);
        assertTrue(usage.servedFrom() != PromptCache.Tier.FRESH, usage.toString());
        assertTrue(usage.promptNanos() > 0 && usage.predictedNanos() > 0, usage.toString());
    }

    @Test
    void requestToolsOverrideTheWeldedDefault() {
        // request > view default: different tools serve correctly (byte-identical to the base
        // model offered the same conversation), uncached, with ONE stderr warning per view
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of(RESET));
        ToolSpecification other =
                ToolSpecification.builder()
                        .name("get_time")
                        .description("Get the current local time")
                        .parameters(JsonObjectSchema.builder().build())
                        .build();
        String question = "Where do I reset my password?";
        // pinned seed: the byte-identity comparison below must not ride on sampling luck
        var params =
                JinferChatRequestParameters.builder().toolSpecifications(other).seed(7L).build();
        var err = new ByteArrayOutputStream();
        PrintStream real = System.err;
        ChatResponse overridden;
        try {
            System.setErr(new PrintStream(err, true));
            overridden =
                    support.chat(
                            ChatRequest.builder()
                                    .messages(UserMessage.from(question))
                                    .parameters(params)
                                    .build());
            support.chat(
                    ChatRequest.builder()
                            .messages(UserMessage.from(question))
                            .parameters(params)
                            .build());
        } finally {
            System.setErr(real);
        }
        String warnings = err.toString();
        assertTrue(warnings.contains("open_reset_portal"), warnings);
        assertTrue(warnings.contains("get_time"), warnings);
        assertEquals(
                warnings.indexOf("WARNING"),
                warnings.lastIndexOf("WARNING"),
                "the override warns ONCE per view: " + warnings);

        ChatResponse plain =
                base.chat(
                        ChatRequest.builder()
                                .messages(SUPPORT.get(0), UserMessage.from(question))
                                .parameters(params)
                                .build());
        assertEquals(plain.aiMessage().text(), overridden.aiMessage().text());
        assertEquals(
                plain.aiMessage().toolExecutionRequests(),
                overridden.aiMessage().toolExecutionRequests());
    }

    @Test
    void toolChoiceNoneOverridesTheWeldedDefault() {
        // NONE empties the effective tool offer - on a tooled view that is an override like any
        // other: served without tools, never a rejection
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of(RESET));
        ChatResponse r =
                support.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Open the reset portal for bob@example.com."))
                                .toolChoice(ToolChoice.NONE)
                                .build());
        assertTrue(!r.aiMessage().hasToolExecutionRequests(), r.aiMessage().toString());
        assertTrue(!r.aiMessage().text().isBlank());
    }

    /** The mainstream idiom this feature exists for. */
    static class Portal {
        final AtomicBoolean opened = new AtomicBoolean();

        @Tool("Open the password reset portal for a customer email")
        String openResetPortal(String email) {
            opened.set(true);
            return "portal opened for " + email;
        }
    }

    interface Assistant {
        String chat(String message);
    }

    @Test
    void aiServicesToolAgentRidesTheView() {
        // AiServices re-sends its @Tool specifications on every request; welding the SAME set
        // makes each of those requests a cache hit - this must serve, never reject
        Portal portal = new Portal();
        JinferChatModel support =
                base.withCachedPrompt(SUPPORT, ToolSpecifications.toolSpecificationsFrom(portal));
        Assistant assistant =
                AiServices.builder(Assistant.class).chatModel(support).tools(portal).build();
        String answer =
                assistant.chat("Please open the reset portal for bob@example.com using the tool.");
        assertTrue(answer != null && !answer.isBlank());
        // WHETHER the model calls is model behavior; the wire contract is that the view served
        Assumptions.assumeTrue(portal.opened.get(), "model chose not to call the tool: " + answer);
    }

    @Test
    void wrongModelArtifactFailsAtBuild() throws Exception {
        Path artifact = Files.createTempDirectory("cached-prompts").resolve("lfm2.jkv");
        base.withCachedPrompt(SUPPORT, List.of());
        base.saveCachedPrompts(artifact);

        Path other =
                TestModels.require(
                        "hf.co/unsloth/gemma-4-E2B-it-qat-GGUF/gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf");
        assertThrows(
                Exception.class,
                () ->
                        JinferChatModel.builder()
                                .modelPath(other)
                                .contextLength(2048)
                                .promptCache(artifact)
                                .build());
    }
}
