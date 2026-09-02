package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ChatRequestParameters;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.output.FinishReason;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The supported two-phase workflow: finish the tool round, then remove tools and constrain the
 * final answer to the JSON schema.
 */
@Tag("integration")
class ToolsWithSchemaIT {

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf"))
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .temperature(0.0)
                        .seed(7L)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    static final ToolSpecification WEATHER =
            ToolSpecification.builder()
                    .name("get_weather")
                    .description("Current weather for a city")
                    .parameters(
                            JsonObjectSchema.builder()
                                    .addStringProperty("city")
                                    .required("city")
                                    .build())
                    .build();

    static ChatRequestParameters toolsOnly() {
        return ChatRequestParameters.builder().toolSpecifications(WEATHER).build();
    }

    static ChatRequestParameters schemaOnly() {
        return ChatRequestParameters.builder()
                .responseFormat(
                        ResponseFormat.builder()
                                .type(ResponseFormatType.JSON)
                                .jsonSchema(
                                        JsonSchema.builder()
                                                .name("Weather")
                                                .rootElement(
                                                        JsonObjectSchema.builder()
                                                                .addStringProperty("city")
                                                                .addNumberProperty("temperature_c")
                                                                .required("city", "temperature_c")
                                                                .build())
                                                .build())
                                .build())
                .build();
    }

    @Test
    void toolRoundThenSchemaShapesTheAnswer() {
        UserMessage user =
                UserMessage.from(
                        "What is the weather in Munich right now? Check with your tool - do not"
                                + " guess.");
        var r1 = model.chat(ChatRequest.builder().messages(user).parameters(toolsOnly()).build());
        AiMessage ai = r1.aiMessage();
        assertEquals(1, ai.toolExecutionRequests().size(), "the mask must not trap the call");
        assertEquals("get_weather", ai.toolExecutionRequests().get(0).name());
        assertEquals(FinishReason.TOOL_EXECUTION, r1.finishReason());

        var r2 =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        user,
                                        ai,
                                        ToolExecutionResultMessage.from(
                                                ai.toolExecutionRequests().get(0), "18C, sunny"))
                                .parameters(schemaOnly())
                                .build());
        String text = r2.aiMessage().text().strip();
        Object parsed = Json.parse(text);
        assertTrue(parsed instanceof Map, "schema-shaped answer expected: " + text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertTrue(String.valueOf(map.get("city")).contains("Munich"), "grounded: " + text);
        assertEquals(18.0, ((Number) map.get("temperature_c")).doubleValue(), 0.01, text);
        assertEquals(FinishReason.STOP, r2.finishReason());
    }
}
