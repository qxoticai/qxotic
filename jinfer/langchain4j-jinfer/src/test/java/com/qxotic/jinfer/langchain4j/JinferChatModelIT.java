package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * End-to-end against a real GGUF (LFM2: native template port, tool-capable). Model-gated:
 * assume-skips when the file is absent. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl langchain4j-jinfer}
 */
@Tag("integration")
class JinferChatModelIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel", ModelFixture.LFM25_8B_Q8.path().toString()));

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
                                .toolChoice(dev.langchain4j.model.chat.request.ToolChoice.NONE)
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
        PersonExtractor extractor =
                dev.langchain4j.service.AiServices.create(PersonExtractor.class, model);
        Person p = extractor.extract("Johann is 42 years old and lives in Munich.");
        assertEquals("Johann", p.name());
        assertEquals(42, p.age());
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
