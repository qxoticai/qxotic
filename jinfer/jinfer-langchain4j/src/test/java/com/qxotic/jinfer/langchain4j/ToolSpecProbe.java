package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Diagnostic (not a gate): does the AiServices tools battery fail because the provider mishandles
 * tools, or because the kit's specs are degenerate? The kit compiles without {@code -parameters},
 * so its @Tool methods reach the model as {@code add(arg0, arg1)} with NO description anywhere.
 * This asks the SAME model the SAME question with the bare spec and with a described one.
 */
@Tag("driver")
@EnabledIf("com.qxotic.jinfer.langchain4j.ToolSpecProbe#modelAvailable")
class ToolSpecProbe {

    static boolean modelAvailable() {
        return TestModels.find(JinferChatModelTckIT.REF).isPresent();
    }

    @Test
    void bareVersusDescribedToolSpec() {
        try (JinferChatModel model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                        .contextLength(8192)
                        .maxOutputTokens(512)
                        .temperature(0.0)
                        .thinking(JinferChatModelTckIT.tckThinking())
                        .seed(7L)
                        .build()) {

            // exactly what the kit sends: no tool description, no parameter descriptions, and
            // parameter names the compiler invented
            ToolSpecification bare =
                    ToolSpecification.builder()
                            .name("add")
                            .parameters(
                                    JsonObjectSchema.builder()
                                            .addProperty("arg0", new JsonIntegerSchema())
                                            .addProperty("arg1", new JsonIntegerSchema())
                                            .required("arg0", "arg1")
                                            .build())
                            .build();

            // the same tool, described the way an application would write it
            ToolSpecification described =
                    ToolSpecification.builder()
                            .name("add")
                            .description("Adds two integers and returns their sum")
                            .parameters(
                                    JsonObjectSchema.builder()
                                            .addIntegerProperty("a", "the first addend")
                                            .addIntegerProperty("b", "the second addend")
                                            .required("a", "b")
                                            .build())
                            .build();

            for (var spec : List.of(bare, described)) {
                ChatResponse r =
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("How much is 37 plus 87?"))
                                        .toolSpecifications(spec)
                                        .build());
                System.out.println(
                        "\n=== spec: "
                                + (spec == bare ? "BARE (the kit's)" : "DESCRIBED")
                                + "\n    calls: "
                                + r.aiMessage().toolExecutionRequests()
                                + "\n    text : "
                                + abbreviate(r.aiMessage().text()));
            }
        }
    }

    /**
     * The tools + response-schema seam. A WELL-DESCRIBED tool the model demonstrably calls (see
     * above), asked the same question twice: once with tools alone, once with tools AND a JSON
     * response format. If the call disappears when the schema arrives, the schema is winning the
     * dispatch point rather than the model changing its mind.
     */
    @Test
    void toolsWithAndWithoutResponseSchema() {
        try (JinferChatModel model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                        .contextLength(8192)
                        .maxOutputTokens(512)
                        .temperature(0.0)
                        .thinking(JinferChatModelTckIT.tckThinking())
                        .seed(7L)
                        .build()) {

            ToolSpecification weather =
                    ToolSpecification.builder()
                            .name("getWeather")
                            .description("Returns the current weather for a city")
                            .parameters(
                                    JsonObjectSchema.builder()
                                            .addStringProperty("city", "the city to look up")
                                            .required("city")
                                            .build())
                            .build();

            var schema =
                    dev.langchain4j.model.chat.request.ResponseFormat.builder()
                            .type(dev.langchain4j.model.chat.request.ResponseFormatType.JSON)
                            .jsonSchema(
                                    dev.langchain4j.model.chat.request.json.JsonSchema.builder()
                                            .name("weather")
                                            .rootElement(
                                                    JsonObjectSchema.builder()
                                                            .addStringProperty("summary")
                                                            .required("summary")
                                                            .build())
                                            .build())
                            .build();

            ChatRequest.Builder base =
                    ChatRequest.builder()
                            .messages(UserMessage.from("What is the weather in Munich?"))
                            .toolSpecifications(weather);

            ChatResponse toolsOnly = model.chat(base.build());
            System.out.println(
                    "\n=== tools ONLY (control)"
                            + "\n    calls: "
                            + toolsOnly.aiMessage().toolExecutionRequests()
                            + "\n    text : "
                            + abbreviate(toolsOnly.aiMessage().text()));

            ChatResponse both = model.chat(base.responseFormat(schema).build());
            System.out.println(
                    "\n=== tools + RESPONSE SCHEMA"
                            + "\n    calls: "
                            + both.aiMessage().toolExecutionRequests()
                            + "\n    text : "
                            + abbreviate(both.aiMessage().text())
                            + "\n    think: "
                            + abbreviate(both.aiMessage().thinking()));
        }
    }

    private static String abbreviate(String s) {
        if (s == null) return "null";
        String one = s.replace('\n', ' ').strip();
        return one.length() <= 160 ? one : one.substring(0, 160) + "...";
    }
}
