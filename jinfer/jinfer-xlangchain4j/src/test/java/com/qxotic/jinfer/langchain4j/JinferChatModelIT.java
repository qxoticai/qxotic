package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import dev.langchain4j.model.chat.listener.ChatModelResponseContext;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ToolChoice;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import dev.langchain4j.service.AiServices;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end against a real GGUF (LFM2: native template port, tool-capable). Model-gated:
 * assume-skips when the file is absent. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-langchain4j}
 */
@Tag("integration")
class JinferChatModelIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel",
                            TestModels.find(
                                            "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf")
                                    .orElse(
                                            Path.of(
                                                    "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf"))
                                    .toString()));

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void blockingChat() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Answer with exactly one word: what is the capital"
                                                        + " of France?"))
                                .build());
        assertNotNull(r.aiMessage().text());
        assertTrue(r.aiMessage().text().contains("Paris"), r.aiMessage().text());
        assertEquals(FinishReason.STOP, r.finishReason());
        assertTrue(r.tokenUsage().inputTokenCount() > 0);
        assertTrue(r.tokenUsage().outputTokenCount() > 0);
    }

    @Test
    void streamingChat() throws Exception {
        var done = new CompletableFuture<ChatResponse>();
        StringBuilder streamed = new StringBuilder();
        model.streaming()
                .chat(
                        ChatRequest.builder()
                                .messages(UserMessage.from("Count from 1 to 5, digits only."))
                                .build(),
                        new StreamingChatResponseHandler() {
                            @Override
                            public void onPartialResponse(String partial) {
                                streamed.append(partial);
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
        // the streamed fragments and the final message agree
        assertEquals(r.aiMessage().text(), streamed.toString());
    }

    @Test
    void toolRoundTrip() {
        ToolSpecification weather =
                ToolSpecification.builder()
                        .name("get_weather")
                        .description("Get current weather for a city")
                        .parameters(
                                JsonObjectSchema.builder()
                                        .addStringProperty("city")
                                        .required("city")
                                        .build())
                        .build();
        ChatRequest ask =
                ChatRequest.builder()
                        .messages(UserMessage.from("What is the weather in Paris? Use the tool."))
                        .toolSpecifications(weather)
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
                                .toolSpecifications(weather)
                                .build());
        assertNotNull(second.aiMessage().text());
        assertTrue(second.aiMessage().text().contains("18"), second.aiMessage().text());

        // toolChoice NONE: same tool, same call-inviting prompt - the tool is never offered,
        // so the model cannot call it
        ChatResponse none =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the weather in Paris? Use the tool."))
                                .toolSpecifications(weather)
                                .toolChoice(ToolChoice.NONE)
                                .build());
        assertTrue(
                !none.aiMessage().hasToolExecutionRequests(),
                "NONE must prevent tool calls: " + none.aiMessage());
    }

    record Person(String name, int age) {}

    interface PersonExtractor {
        Person extract(String text);
    }

    @Test
    void aiServicesStructuredExtraction() {
        // supportedCapabilities() advertises RESPONSE_FORMAT_JSON_SCHEMA, so AiServices uses the
        // grammar-constrained schema path for the POJO - not prompt-based JSON begging
        PersonExtractor extractor = AiServices.create(PersonExtractor.class, model);
        Person p = extractor.extract("Johann is 42 years old and lives in Munich.");
        assertEquals("Johann", p.name());
        assertEquals(42, p.age());
    }

    @Test
    void maxOutputTokensTruncatesWithLengthFinish() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Count from 1 to 100, digits and spaces only."))
                                .maxOutputTokens(20)
                                .build());
        assertEquals(FinishReason.LENGTH, r.finishReason());
        assertEquals(20, r.tokenUsage().outputTokenCount());
    }

    @Test
    void stopSequencesTruncateContent() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Count from 1 to 20 as plain digits separated by"
                                                        + " spaces."))
                                .stopSequences(List.of("5"))
                                .build());
        assertEquals(FinishReason.STOP, r.finishReason());
        assertTrue(!r.aiMessage().text().contains("5"), r.aiMessage().text());
    }

    @Test
    void thinkingExposedAndSuppressible() {
        ChatResponse thinking = model.chat(UserMessage.from("What is 17 + 25? Answer briefly."));
        Assumptions.assumeTrue(
                thinking.aiMessage().thinking() != null, "not a thinking model reply");
        assertTrue(!thinking.aiMessage().thinking().isBlank());
        assertTrue(!thinking.aiMessage().text().isBlank());

        try (JinferChatModel quiet =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(2048)
                        .maxOutputTokens(128)
                        .thinking(false)
                        .build()) {
            ChatResponse plain = quiet.chat(UserMessage.from("What is 17 + 25? Answer briefly."));
            assertEquals(null, plain.aiMessage().thinking());
            assertTrue(!plain.aiMessage().text().isBlank());
        }
    }

    @Test
    void chatModelListenerReceivesRequestAndResponse() {
        var seen = new AtomicReference<ChatResponse>();
        var listener =
                new ChatModelListener() {
                    @Override
                    public void onResponse(ChatModelResponseContext ctx) {
                        seen.set(ctx.chatResponse());
                    }
                };
        try (JinferChatModel listened =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"))
                        .contextLength(1024)
                        .maxOutputTokens(32)
                        .listeners(List.of(listener))
                        .build()) {
            listened.chat(UserMessage.from("Say hi."));
            assertNotNull(seen.get(), "listener saw the response");
            assertTrue(seen.get().tokenUsage().totalTokenCount() > 0);
            assertNotNull(seen.get().finishReason());
        }
    }

    @Test
    void multiTurn() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from("Remember this codeword: PELICAN."),
                                        AiMessage.from("Understood, I will remember it."),
                                        UserMessage.from("What was the codeword? One word only."))
                                .build());
        assertTrue(r.aiMessage().text().contains("PELICAN"), r.aiMessage().text());
    }
}
