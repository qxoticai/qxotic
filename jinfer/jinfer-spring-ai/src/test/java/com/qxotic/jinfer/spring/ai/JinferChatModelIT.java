package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import io.micrometer.observation.tck.TestObservationRegistry;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.metadata.EmptyRateLimit;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * End-to-end against a real GGUF (LFM2: native template port, tool-capable). Model-gated:
 * assume-skips when the file is absent. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class JinferChatModelIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel", ModelFixture.LFM25_8B_Q8.path().toString()));

    static JinferChatModel model;
    static TestObservationRegistry observations;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        observations = TestObservationRegistry.create();
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(512)
                        .observationRegistry(observations)
                        .build();
    }

    @Test
    void blockingChat() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage(
                                        "Answer with exactly one word: what is the capital of"
                                                + " France?")));
        String text = r.getResult().getOutput().getText();
        assertNotNull(text);
        assertTrue(text.contains("Paris"), text);
        assertEquals("stop", r.getResult().getMetadata().getFinishReason());
        assertTrue(r.getMetadata().getUsage().getPromptTokens() > 0);
        assertTrue(r.getMetadata().getUsage().getCompletionTokens() > 0);
    }

    @Test
    void toolCallIsReturnedNotExecuted() {
        ToolDefinition def =
                DefaultToolDefinition.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .inputSchema(
                                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}")
                        .build();
        ToolCallback weather =
                new ToolCallback() {
                    @Override
                    public ToolDefinition getToolDefinition() {
                        return def;
                    }

                    @Override
                    public String call(String toolInput) {
                        throw new AssertionError("2.0 models never execute tools");
                    }
                };
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("What is the weather in Paris?"),
                                JinferChatOptions.builder()
                                        .toolCallbacks(List.of(weather))
                                        .maxTokens(512)
                                        .build()));
        AssistantMessage ai = r.getResult().getOutput();
        assertTrue(ai.hasToolCalls(), "expected tool calls, got: " + ai.getText());
        assertEquals("get_weather", ai.getToolCalls().get(0).name());
        assertTrue(
                ai.getToolCalls().get(0).arguments().contains("Paris"),
                ai.getToolCalls().get(0).arguments());
        assertEquals("tool_calls", r.getResult().getMetadata().getFinishReason());
        assertTrue(r.hasToolCalls());
    }

    @Test
    void streamingConsistentWithBlocking() {
        // a short, tie-free answer: two greedy passes can diverge at logit ties on long replies,
        // so this test needs a prompt whose reply is stable across independent passes
        Prompt prompt =
                new Prompt(
                        new UserMessage(
                                "Answer with exactly one word: what is the capital of France?"));
        String blocking = model.call(prompt).getResult().getOutput().getText();
        List<ChatResponse> chunks = model.stream(prompt).collectList().block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        StringBuilder streamed = new StringBuilder();
        for (ChatResponse c : chunks) {
            AssistantMessage out = c.getResult().getOutput();
            // the content lane only: reasoning streams as isThought chunks, blocking puts it in
            // metadata
            if (!Boolean.TRUE.equals(out.getMetadata().get(JinferChatModel.IS_THOUGHT_KEY))) {
                streamed.append(out.getText());
            }
        }
        ChatResponse last = chunks.get(chunks.size() - 1);
        assertEquals(blocking, streamed.toString());
        assertEquals("stop", last.getResult().getMetadata().getFinishReason());
        assertTrue(last.getMetadata().getUsage().getCompletionTokens() > 0);
    }

    @Test
    void toolRoundTripMultiTurn() {
        ToolCallback weather = weatherTool();
        UserMessage question =
                new UserMessage("What is the weather in Paris? Use the get_weather tool.");
        ChatResponse first =
                model.call(
                        new Prompt(
                                question,
                                JinferChatOptions.builder()
                                        .toolCallbacks(List.of(weather))
                                        .maxTokens(512)
                                        .build()));
        AssistantMessage callMessage = first.getResult().getOutput();
        Assumptions.assumeTrue(
                callMessage.hasToolCalls(),
                "model chose not to call the tool: " + callMessage.getText());
        AssistantMessage.ToolCall call = callMessage.getToolCalls().get(0);
        // the manual loop: assistant's call + the tool result go back in, answer grounds on it
        ChatResponse second =
                model.call(
                        new Prompt(
                                List.of(
                                        question,
                                        callMessage,
                                        ToolResponseMessage.builder()
                                                .responses(
                                                        List.of(
                                                                new ToolResponseMessage
                                                                        .ToolResponse(
                                                                        call.id(),
                                                                        call.name(),
                                                                        "18C, sunny")))
                                                .build()),
                                JinferChatOptions.builder()
                                        .toolCallbacks(List.of(weather))
                                        .maxTokens(512)
                                        .build()));
        String answer = second.getResult().getOutput().getText();
        assertNotNull(answer);
        assertTrue(answer.contains("18"), answer);
    }

    @Test
    void streamingToolCallsArriveOnTheFinalChunk() {
        List<ChatResponse> chunks =
                model.stream(
                                new Prompt(
                                        new UserMessage(
                                                "What is the weather in Paris? Use the"
                                                        + " get_weather tool."),
                                        JinferChatOptions.builder()
                                                .toolCallbacks(List.of(weatherTool()))
                                                .maxTokens(512)
                                                .build()))
                        .collectList()
                        .block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        ChatResponse last = chunks.get(chunks.size() - 1);
        Assumptions.assumeTrue(
                last.getResult().getOutput().hasToolCalls(), "model chose not to call the tool");
        // no partial tool streaming: deltas carried nothing, the final chunk carries the call
        AssistantMessage.ToolCall call = last.getResult().getOutput().getToolCalls().get(0);
        assertEquals("get_weather", call.name());
        assertTrue(call.arguments().contains("Paris"), call.arguments());
        assertEquals("tool_calls", last.getResult().getMetadata().getFinishReason());
    }

    private static ToolCallback weatherTool() {
        ToolDefinition def =
                DefaultToolDefinition.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .inputSchema(
                                "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}")
                        .build();
        return new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
                return def;
            }

            @Override
            public String call(String toolInput) {
                return "18C, sunny";
            }
        };
    }

    @Test
    void rejectsUnsupportedKnobsSynchronously() {
        // request-shape errors throw from call() itself, never on a worker thread
        var e1 =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                new UserMessage("hi"),
                                                JinferChatOptions.builder().topK(10).build())));
        assertEquals("topK is not supported", e1.getMessage());
        var e2 =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                new UserMessage("hi"),
                                                JinferChatOptions.builder()
                                                        .frequencyPenalty(0.5)
                                                        .build())));
        assertEquals("frequencyPenalty is not supported", e2.getMessage());
        var e3 =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                new UserMessage("hi"),
                                                JinferChatOptions.builder()
                                                        .timeout(Duration.ofSeconds(-1))
                                                        .build())));
        assertEquals("timeout must not be negative", e3.getMessage());
    }

    @Test
    void rejectsPerRequestModelSwitch() {
        var e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                new UserMessage("hi"),
                                                JinferChatOptions.builder()
                                                        .model("some-other.gguf")
                                                        .build())));
        assertTrue(e.getMessage().contains("one loaded GGUF per instance"), e.getMessage());
    }

    @Test
    void intermediateStreamChunksCarryNoMetadata() {
        List<ChatResponse> chunks =
                model.stream(new Prompt(new UserMessage("Count from 1 to 9, digits only.")))
                        .collectList()
                        .block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        assertTrue(chunks.size() > 2, "expected several chunks, got " + chunks.size());
        // only the final chunk carries a finish reason and usage
        for (ChatResponse c : chunks.subList(0, chunks.size() - 1)) {
            assertNull(
                    c.getResult().getMetadata().getFinishReason(),
                    "intermediate chunk has a finish reason");
        }
        ChatResponse last = chunks.get(chunks.size() - 1);
        assertNotNull(last.getResult().getMetadata().getFinishReason());
        assertTrue(last.getMetadata().getUsage().getCompletionTokens() > 0);
    }

    @Test
    void perRequestMaxTokensTruncates() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("Count from 1 to 100, digits and spaces only."),
                                JinferChatOptions.builder().maxTokens(20).build()));
        assertEquals("length", r.getResult().getMetadata().getFinishReason());
        assertEquals(Integer.valueOf(20), r.getMetadata().getUsage().getCompletionTokens());
    }

    @Test
    void thinkingExposedInMetadataAndSuppressible() {
        ChatResponse thinking =
                model.call(new Prompt(new UserMessage("What is 17 + 25? Answer briefly.")));
        Assumptions.assumeTrue(
                thinking.getResult().getOutput().getMetadata().get("thinking") != null,
                "not a thinking model reply");
        ChatResponse suppressed =
                model.call(
                        new Prompt(
                                new UserMessage("What is 17 + 25? Answer briefly."),
                                JinferChatOptions.builder().thinking(false).build()));
        assertNull(suppressed.getResult().getOutput().getMetadata().get("thinking"));
        assertNotNull(suppressed.getResult().getOutput().getText());
    }

    @Test
    void stopSequences() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("Count from 1 to 9, digits and spaces only."),
                                JinferChatOptions.builder()
                                        .stopSequences(List.of("5"))
                                        .maxTokens(512)
                                        .build()));
        String text = r.getResult().getOutput().getText();
        assertNotNull(text);
        assertTrue(!text.contains("5"), text);
        assertEquals("stop", r.getResult().getMetadata().getFinishReason());
    }

    @Test
    void usageMetadataIsComplete() {
        ChatResponse r = model.call(new Prompt(new UserMessage("One word: ok?")));
        var usage = r.getMetadata().getUsage();
        // native usage: the exact phase timings of the pass
        var nativeUsage =
                assertInstanceOf(JinferChatModel.JinferUsage.class, usage.getNativeUsage());
        assertTrue(nativeUsage.promptNanos() > 0);
        assertTrue(nativeUsage.predictedNanos() > 0);
        // Ollama-style timing key-values
        assertNotNull(r.getMetadata().get("prompt-eval-duration"));
        assertNotNull(r.getMetadata().get("eval-duration"));
        // no cache involved on the base model: null, not zero
        assertNull(usage.getCacheReadInputTokens());
        assertNull(usage.getCacheWriteInputTokens());
        // no quota exists in-process, but the slot must not be null
        assertInstanceOf(EmptyRateLimit.class, r.getMetadata().getRateLimit());
    }

    @Test
    void observationOnCall() {
        observations.clear();
        model.call(new Prompt(new UserMessage("One word: ok?")));
        observations
                .assertThat()
                .hasNumberOfObservationsWithNameEqualTo("gen_ai.client.operation", 1);
        observations.assertThat().hasAnObservationWithAKeyValue("gen_ai.system", "jinfer");
        observations
                .assertThat()
                .hasAnObservationWithAKeyValue(
                        "gen_ai.request.model", MODEL.getFileName().toString());
        // (the convention emits gen_ai.request.stream only on streaming spans)
        // the convention read the response: usage and finish reason made it onto the span
        observations.assertThat().hasAnObservationWithAKeyName("gen_ai.usage.input_tokens");
        observations.assertThat().hasAnObservationWithAKeyName("gen_ai.usage.output_tokens");
        observations.assertThat().hasAnObservationWithAKeyName("gen_ai.response.finish_reasons");
    }

    @Test
    void observationOnStream() {
        observations.clear();
        model.stream(new Prompt(new UserMessage("One word: ok?")))
                .collectList()
                .block(Duration.ofMinutes(2));
        observations
                .assertThat()
                .hasNumberOfObservationsWithNameEqualTo("gen_ai.client.operation", 1);
        observations.assertThat().hasAnObservationWithAKeyValue("gen_ai.request.stream", "true");
        observations.assertThat().hasAnObservationWithAKeyName("gen_ai.usage.input_tokens");
    }

    @Test
    void streamFluxIsResubscribable() {
        observations.clear();
        var flux =
                model.stream(
                        new Prompt(
                                new UserMessage("One word: ok?"),
                                JinferChatOptions.builder().maxTokens(8).build()));
        flux.collectList().block(Duration.ofMinutes(2));
        flux.collectList().block(Duration.ofMinutes(2));
        // each subscription gets its own observation (no shared-Observation race)
        observations
                .assertThat()
                .hasNumberOfObservationsWithNameEqualTo("gen_ai.client.operation", 2);
    }

    @Test
    void observationRecordsValidationErrors() {
        observations.clear();
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        model.call(
                                new Prompt(
                                        new UserMessage("hi"),
                                        JinferChatOptions.builder().topK(10).build())));
        observations
                .assertThat()
                .hasHandledContextsThatSatisfy(
                        contexts -> {
                            assertEquals(1, contexts.size());
                            assertInstanceOf(
                                    IllegalArgumentException.class, contexts.get(0).getError());
                        });
    }

    @Test
    void streamingThinkingChunksAreFlagged() {
        List<ChatResponse> chunks =
                model.stream(new Prompt(new UserMessage("What is 17 + 25? Answer briefly.")))
                        .collectList()
                        .block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        StringBuilder thoughts = new StringBuilder();
        StringBuilder content = new StringBuilder();
        for (ChatResponse c : chunks.subList(0, chunks.size() - 1)) {
            AssistantMessage out = c.getResult().getOutput();
            if (Boolean.TRUE.equals(out.getMetadata().get(JinferChatModel.IS_THOUGHT_KEY))) {
                thoughts.append(out.getText());
            } else {
                content.append(out.getText());
            }
        }
        Assumptions.assumeTrue(!thoughts.isEmpty(), "not a thinking model reply");
        assertTrue(!content.isEmpty(), "no content streamed");
        // the final chunk's thinking metadata is the same reasoning the thought chunks streamed
        String thinking =
                (String)
                        chunks.get(chunks.size() - 1)
                                .getResult()
                                .getOutput()
                                .getMetadata()
                                .get("thinking");
        assertEquals(thinking, thoughts.toString());
    }

    @Test
    void streamingThinkingOffEmitsNoThoughtChunks() {
        List<ChatResponse> chunks =
                model.stream(
                                new Prompt(
                                        new UserMessage("What is 17 + 25? Answer briefly."),
                                        JinferChatOptions.builder().thinking(false).build()))
                        .collectList()
                        .block(Duration.ofMinutes(2));
        assertNotNull(chunks);
        for (ChatResponse c : chunks) {
            assertNull(c.getResult().getOutput().getMetadata().get(JinferChatModel.IS_THOUGHT_KEY));
        }
    }
}
