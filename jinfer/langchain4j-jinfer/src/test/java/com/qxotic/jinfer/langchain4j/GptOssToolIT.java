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
import dev.langchain4j.model.output.FinishReason;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Set;
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
