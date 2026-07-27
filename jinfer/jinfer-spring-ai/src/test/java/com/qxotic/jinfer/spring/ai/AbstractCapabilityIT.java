package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * The model-agnostic CAPABILITY contract on the Spring AI surface: tool calling and structured
 * output must behave identically whatever the family's wire syntax (pythonic spans, JSON blocks,
 * XML, Harmony channels). Wire fidelity per family is the langchain4j AbstractToolIT + wire
 * battery's job; THIS contract pins the Spring mapping layer: calls surface as {@link
 * AssistantMessage.ToolCall}s, round-trips ground, schemas bind output only, conflicts are loud.
 *
 * <p>Parameterized by model: each subclass names one GGUF (system-property overridable);
 * assume-skips when absent. Output assertions strip() first (the documented boilerplate-newline
 * law).
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class AbstractCapabilityIT {

    /** The GGUF this subclass runs against. */
    abstract Path modelPath();

    static final String SCHEMA =
            "{\"type\":\"object\",\"properties\":{"
                    + "\"city\":{\"type\":\"string\"},"
                    + "\"population_millions\":{\"type\":\"number\"}},"
                    + "\"required\":[\"city\",\"population_millions\"]}";

    JinferChatModel model;

    @BeforeAll
    void load() {
        Assumptions.assumeTrue(Files.exists(modelPath()), "model not found: " + modelPath());
        model =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .contextLength(4096)
                        .maxTokens(512)
                        .build();
    }

    @AfterAll
    void unload() {
        if (model != null) model.close();
    }

    static ToolCallback weatherTool() {
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
    void toolCallSurfacesWithNameAndArguments() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage(
                                        "What is the weather in Paris? Use the get_weather tool."),
                                JinferChatOptions.builder()
                                        .toolCallbacks(List.of(weatherTool()))
                                        .build()));
        AssistantMessage out = r.getResult().getOutput();
        Assumptions.assumeTrue(out.hasToolCalls(), "model chose not to call: " + out.getText());
        AssistantMessage.ToolCall call = out.getToolCalls().get(0);
        assertEquals("get_weather", call.name());
        Object args = JsonCodec.parse(call.arguments());
        assertInstanceOf(Map.class, args, call.arguments());
        assertTrue(
                String.valueOf(((Map<?, ?>) args).get("city")).toLowerCase().contains("paris"),
                call.arguments());
    }

    @Test
    void toolRoundTripGroundsTheAnswer() {
        UserMessage question =
                new UserMessage("What is the weather in Paris? Use the get_weather tool.");
        ChatResponse first =
                model.call(
                        new Prompt(
                                question,
                                JinferChatOptions.builder()
                                        .toolCallbacks(List.of(weatherTool()))
                                        .build()));
        AssistantMessage callMessage = first.getResult().getOutput();
        Assumptions.assumeTrue(
                callMessage.hasToolCalls(), "model chose not to call: " + callMessage.getText());
        AssistantMessage.ToolCall call = callMessage.getToolCalls().get(0);
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
                                        .toolCallbacks(List.of(weatherTool()))
                                        .build()));
        String answer = second.getResult().getOutput().getText();
        assertNotNull(answer);
        assertTrue(answer.contains("18"), answer);
    }

    @Test
    void outputSchemaShapesTheReply() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage(
                                        "What is the capital of France and roughly how many"
                                                + " million people live in the city?"),
                                JinferChatOptions.builder().outputSchema(SCHEMA).build()));
        String text = r.getResult().getOutput().getText().strip();
        Object parsed;
        try {
            parsed = JsonCodec.parse(text);
        } catch (RuntimeException e) {
            throw new AssertionError(
                    "reply is not valid JSON (" + text.length() + " chars): <<<" + text + ">>>", e);
        }
        assertInstanceOf(Map.class, parsed, text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertEquals(2, map.size(), "no extra keys: " + text);
        assertTrue(String.valueOf(map.get("city")).toLowerCase().contains("paris"), text);
        assertInstanceOf(Number.class, map.get("population_millions"), text);
    }

    @Test
    void schemaAndToolsRejectLoudly() {
        assertThrows(
                RuntimeException.class,
                () ->
                        model.call(
                                new Prompt(
                                        new UserMessage("hi"),
                                        JinferChatOptions.builder()
                                                .outputSchema(SCHEMA)
                                                .toolCallbacks(List.of(weatherTool()))
                                                .build())));
    }
}
