package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.testkit.ModelFixture;
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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Tools AND a JSON-schema response format on ONE request, through the public API: the schema rides
 * the family's reply language, so round 1 still CALLS the tool (the mask admits the family's own
 * call syntax) and round 2's answer can only be schema JSON. What hosted APIs promise as trained
 * behavior, as a mask guarantee.
 */
@Tag("integration")
class ToolsWithSchemaIT {

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(ModelFixture.LFM25_8B_Q8.present(), "model not found");
        model =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_8B_Q8.path())
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

    static ChatRequestParameters toolsAndSchema() {
        return ChatRequestParameters.builder()
                .toolSpecifications(WEATHER)
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
    void theSchemaMaskStillAdmitsTheCallAndThenShapesTheAnswer() {
        // the composed mask deliberately admits BOTH calling and answering - that choice is the
        // model's; the prompt makes it unambiguous so the test pins the MASK, not a near-tie
        UserMessage user =
                UserMessage.from(
                        "What is the weather in Munich right now? Check with your tool - do not"
                                + " guess.");
        var r1 =
                model.chat(
                        ChatRequest.builder().messages(user).parameters(toolsAndSchema()).build());
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
                                .parameters(toolsAndSchema())
                                .build());
        String text = r2.aiMessage().text().strip();
        Object parsed = JsonCodec.parse(text);
        assertTrue(parsed instanceof Map, "schema-shaped answer expected: " + text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertTrue(String.valueOf(map.get("city")).contains("Munich"), "grounded: " + text);
        assertEquals(18.0, ((Number) map.get("temperature_c")).doubleValue(), 0.01, text);
        assertEquals(FinishReason.STOP, r2.finishReason());
    }
}
