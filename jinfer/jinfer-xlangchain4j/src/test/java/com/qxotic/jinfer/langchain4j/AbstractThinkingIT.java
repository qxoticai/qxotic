package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolExecutionRequest;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.PartialThinking;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The model-agnostic REASONING-LANE contract, end-to-end against a real GGUF: thinking streams
 * before content and never leaks markers into it, the lanes reassemble exactly, and a tool round
 * trip through a thinking turn frames cleanly. Whether a turn reasons at all is model behavior
 * (assumption-gated); WHERE reasoning may live is wire law (asserted).
 *
 * <p>Note what is deliberately NOT pinned: langchain4j's Ollama/Anthropic suites assert thinking is
 * not re-sent in follow-up requests. jinfer's unmodified echo replays the exact generated wire -
 * thinking tokens included - because that is what makes cache-extension hits byte-exact (pinned in
 * {@code PromptCacheReuseTest}); the token estimator's reasoning-free counting is pinned in {@code
 * EstimatorsTest}. The two surfaces disagree on purpose.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class AbstractThinkingIT {

    /** The GGUF this subclass runs against - a reasoning-capable family. */
    abstract Path modelPath();

    static final ToolSpecification WEATHER =
            ToolSpecification.builder()
                    .name("get_weather")
                    .description("Get current weather for a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city", "The city name")
                                    .required("city")
                                    .build())
                    .build();

    JinferChatModel model;

    @BeforeAll
    void load() {
        Assumptions.assumeTrue(Files.exists(modelPath()), "model not found: " + modelPath());
        model =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .contextLength(4096)
                        // reasoning families spend tokens on analysis before any answer; a tight
                        // budget would die inside the thinking lane (the TCK's gpt-oss lesson)
                        .maxOutputTokens(1024)
                        .thinking(true)
                        .build();
    }

    @AfterAll
    void unload() {
        if (model != null) model.close();
    }

    /** No family's reasoning scaffold may surface as content. */
    static void assertNoThinkingSyntax(String text) {
        assertFalse(
                text.contains("<think") || text.contains("</think"),
                "think markers leaked into content: " + text);
    }

    @Test
    void reasoningStaysInItsLane() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Is 17 a prime number?"))
                                .build());
        AiMessage ai = r.aiMessage();
        assertNotNull(ai.text(), "an answer exists: " + ai);
        assertNoThinkingSyntax(ai.text());
        Assumptions.assumeTrue(
                ai.thinking() != null && !ai.thinking().isBlank(),
                "this turn did not reason (model behavior): " + ai.text());
        assertFalse(
                ai.thinking().contains(ai.text()),
                "the answer must not be duplicated inside the reasoning lane");
    }

    private record Streamed(
            ChatResponse response,
            List<String> events, // "think" | "content" in arrival order
            String thinking,
            String content) {}

    private Streamed stream(String question) throws Exception {
        CompletableFuture<ChatResponse> done = new CompletableFuture<>();
        List<String> events = new CopyOnWriteArrayList<>();
        StringBuilder thinking = new StringBuilder();
        StringBuilder content = new StringBuilder();
        model.streaming()
                .chat(
                        ChatRequest.builder().messages(UserMessage.from(question)).build(),
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialThinking(PartialThinking partial) {
                                events.add("think");
                                thinking.append(partial.text());
                            }

                            @Override
                            public void onPartialResponse(String partial) {
                                events.add("content");
                                content.append(partial);
                            }

                            @Override
                            public void onCompleteResponse(ChatResponse response) {
                                done.complete(response);
                            }

                            @Override
                            public void onError(Throwable error) {
                                done.completeExceptionally(error);
                            }
                        });
        ChatResponse r = done.get(5, TimeUnit.MINUTES);
        return new Streamed(r, events, thinking.toString(), content.toString());
    }

    private static String nullToEmpty(String s) {
        return s == null ? "" : s;
    }

    @Test
    void thinkingStreamsBeforeContentAndReassemblesExactly() throws Exception {
        Streamed s = stream("Is 17 a prime number?");
        // the order law: every thinking delta precedes every content delta (Ollama's InOrder
        // check, langchain4j-free)
        int lastThink = s.events().lastIndexOf("think");
        int firstContent = s.events().indexOf("content");
        assertTrue(
                lastThink < 0 || firstContent < 0 || lastThink < firstContent,
                "thinking and content interleaved: " + s.events());
        // the reassembly law: streamed deltas concatenate to the final message's lanes exactly
        // (null and "" are the same empty lane - Mappings omits empty lanes, a degenerate
        // family reply can still carry one)
        assertEquals(nullToEmpty(s.response().aiMessage().thinking()), s.thinking());
        assertEquals(nullToEmpty(s.response().aiMessage().text()), s.content());
        assertNoThinkingSyntax(s.content());
        Assumptions.assumeTrue(!s.thinking().isBlank(), "this turn did not reason");
    }

    @Test
    void thinkingSurvivesAToolRoundTrip() throws Exception {
        // Anthropic's interleaved thinking+tools scenario, family-agnostic: a reasoning family
        // calling a tool must keep reasoning OUT of the call payload and the echoed history must
        // frame the post-result turn cleanly
        ChatResponse first =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."))
                                .toolSpecifications(WEATHER)
                                .build());
        if (first.aiMessage().thinking() != null) {
            assertNoThinkingSyntax(
                    first.aiMessage().text() == null ? "" : first.aiMessage().text());
        }
        Assumptions.assumeTrue(
                first.aiMessage().hasToolExecutionRequests(),
                "model chose not to call the tool: " + first.aiMessage().text());
        ToolExecutionRequest call = first.aiMessage().toolExecutionRequests().get(0);
        assertEquals("get_weather", call.name());

        ChatResponse second =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."),
                                        first.aiMessage(), // echoed with its reasoning lane
                                        ToolExecutionResultMessage.from(
                                                call.id(), call.name(), "18C, sunny"))
                                .toolSpecifications(WEATHER)
                                .build());
        assertNotNull(second.aiMessage().text(), second.aiMessage().toString());
        assertNoThinkingSyntax(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());
    }
}
