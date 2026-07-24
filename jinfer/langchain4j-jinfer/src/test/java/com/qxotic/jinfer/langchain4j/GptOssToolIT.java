package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.CompleteToolCall;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end Harmony (gpt-oss) tool calling through the native codec: declarations in the developer
 * block, {@code commentary to=functions.*} calls parsed structurally, tool responses folded back,
 * and {@code REQUIRED} forcing via the {@code <|channel|>} seed + name pin. Model-gated:
 * assume-skips when the GGUF is absent.
 */
@Tag("integration")
class GptOssToolIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.gptossModel",
                            "/home/mukel/Desktop/playground/models/unsloth/gpt-oss-20b-Q8_0.gguf"));

    static final ToolSpecification WEATHER =
            ToolSpecification.builder()
                    .name("get_weather")
                    .description("Get current weather for a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city")
                                    .required("city")
                                    .build())
                    .build();

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(768)
                        .build();
    }

    @Test
    void toolRoundTrip() {
        ChatRequest ask =
                ChatRequest.builder()
                        .messages(UserMessage.from("What is the weather in Paris? Use the tool."))
                        .toolSpecifications(WEATHER)
                        .build();
        ChatResponse first = model.chat(ask);
        Assumptions.assumeTrue(
                first.aiMessage().hasToolExecutionRequests(),
                "model chose not to call the tool: " + first.aiMessage().text());
        assertEquals(FinishReason.TOOL_EXECUTION, first.finishReason());
        var call = first.aiMessage().toolExecutionRequests().get(0);
        assertEquals("get_weather", call.name());
        assertTrue(call.arguments().contains("Paris"), call.arguments());

        ChatResponse second =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."),
                                        first.aiMessage(),
                                        ToolExecutionResultMessage.from(
                                                call.id(), call.name(), "18C, sunny"))
                                .toolSpecifications(WEATHER)
                                .build());
        assertNotNull(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());
    }

    /** Streams one request, recording every callback; returns after onComplete/onError. */
    record Streamed(
            ChatResponse response, String text, String thinking, List<CompleteToolCall> calls) {}

    private static Streamed stream(ChatRequest request) throws Exception {
        var done = new CompletableFuture<ChatResponse>();
        StringBuilder text = new StringBuilder();
        StringBuilder thinking = new StringBuilder();
        List<CompleteToolCall> calls = new ArrayList<>();
        model.streaming()
                .chat(
                        request,
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                text.append(partial);
                            }

                            @Override
                            public void onPartialThinking(
                                    dev.langchain4j.model.chat.response.PartialThinking partial) {
                                thinking.append(partial.text());
                            }

                            @Override
                            public void onCompleteToolCall(CompleteToolCall call) {
                                calls.add(call);
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
        return new Streamed(r, text.toString(), thinking.toString(), calls);
    }

    @Test
    void streamingToolRoundTrip() throws Exception {
        ChatRequest ask =
                ChatRequest.builder()
                        .messages(UserMessage.from("What is the weather in Paris? Use the tool."))
                        .toolSpecifications(WEATHER)
                        .build();
        Streamed first = stream(ask);
        Assumptions.assumeTrue(
                first.response().aiMessage().hasToolExecutionRequests(),
                "model chose not to call the tool: " + first.response().aiMessage().text());
        assertEquals(FinishReason.TOOL_EXECUTION, first.response().finishReason());
        var call = first.response().aiMessage().toolExecutionRequests().get(0);
        assertEquals("get_weather", call.name());
        assertTrue(call.arguments().contains("Paris"), call.arguments());
        // the call was announced whole before the response, and its payload never streamed
        assertEquals(1, first.calls().size());
        assertEquals(call, first.calls().get(0).toolExecutionRequest());
        assertTrue(
                !first.text().contains("Paris") && !first.text().contains("{"),
                "call payload must not stream as content: " + first.text());
        assertTrue(!first.thinking().isEmpty(), "Harmony reasoning streams as thinking");

        Streamed second =
                stream(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."),
                                        first.response().aiMessage(),
                                        ToolExecutionResultMessage.from(
                                                call.id(), call.name(), "18C, sunny"))
                                .toolSpecifications(WEATHER)
                                .build());
        // the streamed fragments and the final message agree
        assertEquals(second.response().aiMessage().text(), second.text());
        assertTrue(second.text().contains("18"), second.text());
    }

    @Test
    void streamingRequiredForcesAnOfferedTool() throws Exception {
        Streamed r =
                stream(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Say hello."))
                                .toolSpecifications(WEATHER)
                                .toolChoice(ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                r.response().aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + r.response().aiMessage());
        assertEquals(FinishReason.TOOL_EXECUTION, r.response().finishReason());
        assertEquals(1, r.calls().size());
        assertEquals("get_weather", r.calls().get(0).toolExecutionRequest().name());
        assertTrue(r.text().isEmpty(), "a forced call streams no content: " + r.text());
    }

    @Test
    void requiredForcesAnOfferedTool() {
        ToolSpecification decoy =
                ToolSpecification.builder()
                        .name("get_time")
                        .description("Get the current local time for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Say hello."))
                                .toolSpecifications(WEATHER, decoy)
                                .toolChoice(ToolChoice.REQUIRED)
                                .build());
        assertTrue(
                r.aiMessage().hasToolExecutionRequests(),
                "REQUIRED must force a call: " + r.aiMessage());
        assertEquals(FinishReason.TOOL_EXECUTION, r.finishReason());
        String name = r.aiMessage().toolExecutionRequests().get(0).name();
        assertTrue(
                Set.of("get_weather", "get_time").contains(name),
                "forced call must name an offered tool: " + name);
    }
}
