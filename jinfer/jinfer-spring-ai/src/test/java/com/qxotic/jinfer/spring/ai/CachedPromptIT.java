package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.SystemMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * Cached prompts end-to-end against a real GGUF (LFM2: native template port with a StateCodec).
 * Model-gated: assume-skips when the file is absent. Run: {@code mvn test
 * -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class CachedPromptIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel", ModelFixture.LFM25_8B_Q8.path().toString()));

    static final List<Message> SUPPORT =
            List.of(
                    new SystemMessage(
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
                        .maxTokens(128)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (base != null) base.close();
    }

    @Test
    void forkOfAnOwningModelRefusesWithTheRecipe() {
        IllegalStateException e = assertThrows(IllegalStateException.class, base::fork);
        assertTrue(e.getMessage().contains("Models.load"), e.getMessage());
        assertTrue(e.getMessage().contains("model(loaded)"), e.getMessage());
    }

    @Test
    void twoModelsAreTwoParallelPipelines() throws Exception {
        // thinking off + pinned seeds + a generous budget: the echo assertion must not ride
        // on sampling luck (a 32-token budget failed on a near-miss "BRAZO" at one seed)
        JinferChatModel base =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(2048)
                        .maxTokens(128)
                        .thinking(false)
                        .seed(1L)
                        .build();
        JinferChatModel twin =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(2048)
                        .maxTokens(128)
                        .thinking(false)
                        .seed(2L)
                        .build();
        var pool = Executors.newFixedThreadPool(2);
        try {
            var a =
                    pool.submit(
                            () ->
                                    base.call(new Prompt(new UserMessage("Say exactly: ALPHA")))
                                            .getResult()
                                            .getOutput()
                                            .getText());
            var b =
                    pool.submit(
                            () ->
                                    twin.call(new Prompt(new UserMessage("Say exactly: BRAVO")))
                                            .getResult()
                                            .getOutput()
                                            .getText());
            assertTrue(a.get().contains("ALPHA"), a.get());
            assertTrue(b.get().contains("BRAVO"), b.get());
        } finally {
            pool.shutdown();
            twin.close();
            base.close();
        }
    }

    @Test
    void cachedSessionsMultiTurn() {
        // cachedSessions(1): turn 2 strictly extends turn 1's pooled state - possible only
        // because the echoed reply restores its verbatim ids through REPLY_KEY metadata, so
        // the re-encode is the exact generated tokens (the round-trip law, spring edition)
        // pinned seed: an unlucky random seed can spend the whole budget on a think span
        JinferChatModel warm =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(128)
                        .cachedSessions(1)
                        .seed(7L)
                        .build();
        try {
            UserMessage first =
                    new UserMessage("Remember the codeword PELICAN. Acknowledge briefly.");
            ChatResponse w1 = warm.call(new Prompt(first));
            Prompt secondTurn =
                    new Prompt(
                            first,
                            w1.getResult().getOutput(),
                            new UserMessage("What was the codeword? Answer with one word."));
            ChatResponse hit = warm.call(secondTurn); // strictly extends the pooled turn-1 state
            String stats = warm.engine.sessionStats();
            assertTrue(stats.contains("hits=1"), "turn 2 must reuse turn 1's live state: " + stats);
            assertTrue(
                    hit.getResult().getOutput().getText().contains("PELICAN"),
                    hit.getResult().getOutput().getText());
        } finally {
            warm.close();
        }
    }

    @Test
    void byteIdentityWithUncached() {
        // OWN engine + pinned seed: the law is view-vs-inline on the SAME tree state, and block
        // KV is bit-exact only against the define that produced it - other tests' defines share
        // head blocks computed in different batch shapes, which can drift an ulp and flip an
        // argmax near-tie. Self-contained, the comparison tests the law, not test order.
        JinferChatModel fresh =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(128)
                        .seed(7L)
                        .build();
        try {
            String question = "Where do I reset my password?";
            // uncached: prefix inlined into the request on the BASE model (never uses the tree)
            ChatResponse plain =
                    fresh.call(new Prompt(List.of(SUPPORT.get(0), new UserMessage(question))));
            // cached: same conversation through a view
            JinferChatModel support = fresh.withCachedPrompt(SUPPORT, List.of());
            ChatResponse cached = support.call(new Prompt(new UserMessage(question)));

            assertEquals(
                    plain.getResult().getOutput().getText(),
                    cached.getResult().getOutput().getText());
            assertTrue(
                    cached.getResult().getOutput().getText().contains("acme.example/reset"),
                    cached.getResult().getOutput().getText());
        } finally {
            fresh.close();
        }
    }

    @Test
    void treeIsConsultedAndBaseStaysCold() {
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        support.call(new Prompt(new UserMessage("Hello?")));
        String stats = base.engine.promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
    }

    @Test
    void cachedViewReportsCacheRead() {
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        ChatResponse cached = support.call(new Prompt(new UserMessage("Hello?")));
        Long cacheRead = cached.getMetadata().getUsage().getCacheReadInputTokens();
        assertNotNull(cacheRead, "a view request must report the restored prefix");
        assertTrue(cacheRead > 0, "restored nothing: " + cacheRead);
        assertTrue(
                cacheRead < cached.getMetadata().getUsage().getPromptTokens(),
                "restored must be a strict prefix: read="
                        + cacheRead
                        + " prompt="
                        + cached.getMetadata().getUsage().getPromptTokens());
        // cache writes are never billed to a request, whichever path served it
        assertNull(cached.getMetadata().getUsage().getCacheWriteInputTokens());
        // NOTE the base model is no longer asserted cold: the block tree serves EVERY prompt now
        // (best-effort, budget-bounded), so a plain request may legitimately report reuse of
        // whatever prefix it shares with earlier traffic
    }

    @Test
    void sharedPrefixDedupAndArtifactRoundTrip() throws Exception {
        String common = "You are an assistant for AcmeCloud. Be brief. ";
        JinferChatModel a =
                base.withCachedPrompt(
                        List.of(new SystemMessage(common + "You handle SUPPORT tickets.")),
                        List.of());
        JinferChatModel b =
                base.withCachedPrompt(
                        List.of(new SystemMessage(common + "You handle SALES questions.")),
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
                        .maxTokens(64)
                        .loadCachedPrompts(artifact)
                        .build();
        JinferChatModel a2 =
                base2.withCachedPrompt(
                        List.of(new SystemMessage(common + "You handle SUPPORT tickets.")),
                        List.of());
        String stats = base2.engine.promptStats();
        assertTrue(stats.contains("hits=") && !stats.contains("hits=0 "), stats);
        ChatResponse r = a2.call(new Prompt(new UserMessage("One word: ok?")));
        assertTrue(!r.getResult().getOutput().getText().isBlank());
        base2.close();
    }

    @Test
    void requestToolsOverrideTheWeldedDefault() {
        // request > view default: per-request tools are served (uncached, warned once), so
        // ChatClient/tool-calling flows compose with views instead of exploding
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        ToolDefinition def = DefaultToolDefinition.builder().name("noop").inputSchema("{}").build();
        ToolCallback noop =
                new ToolCallback() {
                    @Override
                    public ToolDefinition getToolDefinition() {
                        return def;
                    }

                    @Override
                    public String call(String toolInput) {
                        return "";
                    }
                };
        ChatResponse r =
                support.call(
                        new Prompt(
                                new UserMessage("Say OK."),
                                JinferChatOptions.builder().toolCallbacks(List.of(noop)).build()));
        assertTrue(r.getResult().getOutput() != null);
    }

    @Test
    void wrongModelArtifactFailsAtBuild() throws Exception {
        Path artifact = Files.createTempDirectory("cached-prompts").resolve("lfm2.jkv");
        base.withCachedPrompt(SUPPORT, List.of());
        base.saveCachedPrompts(artifact);

        Path other = ModelFixture.LFM25_350M_Q8.path();
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
