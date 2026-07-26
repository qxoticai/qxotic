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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
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

    static final List<org.springframework.ai.chat.messages.Message> SUPPORT =
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

    @Test
    void cachedSessionsMultiTurn() {
        // cachedSessions(1): turn 2 strictly extends turn 1's pooled state - possible only
        // because the echoed reply restores its verbatim ids through REPLY_KEY metadata, so
        // the re-encode is the exact generated tokens (the round-trip law, spring edition)
        JinferChatModel warm =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(128)
                        .cachedSessions(1)
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
        String question = "Where do I reset my password?";
        // uncached: prefix inlined into the request on the BASE model (which never uses the tree)
        ChatResponse plain =
                base.call(new Prompt(List.of(SUPPORT.get(0), new UserMessage(question))));
        // cached: same conversation through a view
        JinferChatModel support = base.withCachedPrompt(SUPPORT, List.of());
        ChatResponse cached = support.call(new Prompt(new UserMessage(question)));

        assertEquals(
                plain.getResult().getOutput().getText(), cached.getResult().getOutput().getText());
        assertTrue(
                cached.getResult().getOutput().getText().contains("acme.example/reset"),
                cached.getResult().getOutput().getText());
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
        // the tree is written at define time, never per request
        assertNull(cached.getMetadata().getUsage().getCacheWriteInputTokens());

        ChatResponse plain = base.call(new Prompt(new UserMessage("Hello?")));
        assertNull(
                plain.getMetadata().getUsage().getCacheReadInputTokens(),
                "the base model never touches the tree");
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
    }

    @Test
    void viewRejectsRequestTools() {
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
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        support.call(
                                new Prompt(
                                        new UserMessage("hi"),
                                        JinferChatOptions.builder()
                                                .toolCallbacks(List.of(noop))
                                                .build())));
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
