package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * Native structured output (grammar-constrained decoding) against a real GGUF. Model-gated via
 * {@link TestModels}. Run: {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-xspring-ai}
 */
@Tag("integration")
class StructuredOutputIT {

    static final Path MODEL =
            TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf");

    static final String SCHEMA =
            "{\"type\":\"object\",\"properties\":{"
                    + "\"answer\":{\"type\":\"string\"},"
                    + "\"confidence\":{\"type\":\"number\"}},"
                    + "\"required\":[\"answer\",\"confidence\"]}";

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .options(JinferChatOptions.builder().maxTokens(512).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJson(String text) {
        try {
            return (Map<String, Object>) Json.parse(text);
        } catch (RuntimeException e) {
            throw new AssertionError(
                    "reply is not valid JSON (" + text.length() + " chars): <<<" + text + ">>>", e);
        }
    }

    @Test
    void streamingConformsToSchema() {
        // a user streaming a schema-constrained reply: concatenated content deltas must be the
        // same schema-conforming JSON guarantee the blocking path gives
        StringBuilder text = new StringBuilder();
        model.stream(
                        new Prompt(
                                new UserMessage("What is the capital of France?"),
                                JinferChatOptions.builder().outputSchema(SCHEMA).build()))
                .doOnNext(
                        chunk -> {
                            if (chunk.getResults().isEmpty()) return;
                            var out = chunk.getResult().getOutput();
                            // reasoning streams too, flagged - only CONTENT deltas carry the JSON
                            boolean thought =
                                    Boolean.TRUE.equals(out.getMetadata().get("isThought"));
                            if (!thought && out.getText() != null) {
                                text.append(out.getText());
                            }
                        })
                .blockLast(Duration.ofMinutes(2));
        Map<String, Object> json = parseJson(text.toString());
        assertTrue(json.get("answer").toString().contains("Paris"), text.toString());
        assertInstanceOf(Number.class, json.get("confidence"), text.toString());
    }

    @Test
    void conformsToSchema() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("What is the capital of France?"),
                                JinferChatOptions.builder().outputSchema(SCHEMA).build()));
        String text = r.getResult().getOutput().getText();
        Map<String, Object> json = parseJson(text); // throws unless the whole reply is valid JSON
        assertTrue(json.get("answer").toString().contains("Paris"), text);
        assertInstanceOf(Number.class, json.get("confidence"), text);
        assertEquals(
                2, json.size(), text); // no extra keys: the grammar admits the schema, nothing else
    }

    @Test
    void thinkingThenConforms() {
        // bounded output (integer + short string) keeps this fast; long constrained strings
        // are fine since the grammar's tail-call fix but still cost a full-vocab mask per token
        String schema =
                "{\"type\":\"object\",\"properties\":{"
                        + "\"result\":{\"type\":\"integer\"}},"
                        + "\"required\":[\"result\"]}";
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage("What is 2 + 2? Think step by step."),
                                JinferChatOptions.builder()
                                        .outputSchema(schema)
                                        .maxTokens(1024)
                                        .build()));
        Map<String, Object> json = parseJson(r.getResult().getOutput().getText());
        // the arithmetic itself is knife-edge (batched-prefill numerics after a force-closed
        // think span) - the contract under test is structural: valid JSON, schema-typed value
        assertInstanceOf(Number.class, json.get("result"), json.toString());
        assertEquals(1, json.size(), json.toString());
        // the grammar stayed dormant during the think span: reasoning was produced, then JSON
        Assumptions.assumeTrue(
                r.getResult().getOutput().getMetadata().get("thinking") != null,
                "not a thinking model reply");
    }

    @Test
    void schemaWithToolsComposesAcrossTheCallRoundTrip() {
        // the composed mask deliberately admits BOTH calling and answering (the schema rides the
        // family's reply language) - the choice is the model's; the prompt makes it unambiguous so
        // the test pins the MASK, not a near-tie (mirrors xlangchain4j's ToolsWithSchemaIT)
        ToolDefinition def =
                DefaultToolDefinition.builder()
                        .name("get_weather")
                        .description("Current weather for a city")
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
                        return "18C, sunny";
                    }
                };
        String schema =
                "{\"type\":\"object\",\"properties\":{"
                        + "\"city\":{\"type\":\"string\"},"
                        + "\"temperature_c\":{\"type\":\"number\"}},"
                        + "\"required\":[\"city\",\"temperature_c\"]}";
        JinferChatOptions options =
                JinferChatOptions.builder()
                        .outputSchema(schema)
                        .toolCallbacks(List.of(weather))
                        .maxTokens(512)
                        // the xlang twin pins this deterministic: the mask admits BOTH calling
                        // and answering, so the choice must not ride a sampled near-tie
                        .temperature(0.0)
                        .seed(7L)
                        .build();
        UserMessage user =
                new UserMessage(
                        "What is the weather in Munich right now? Check with your tool - do not"
                                + " guess.\nWhen you can answer, reply with JSON matching this"
                                + " schema, and nothing else:\n"
                                + schema); // this surface never states the schema itself -
        // BeanOutputConverter does; a test must say it the same way, or the model has no
        // reason to pick the (grammar-forced) JSON answer over calling again
        // round 1: the mask must not trap the call
        ChatResponse r1 = model.call(new Prompt(user, options));
        AssistantMessage callMessage = r1.getResult().getOutput();
        Assumptions.assumeTrue(
                callMessage.hasToolCalls(),
                "model chose to answer directly: " + callMessage.getText());
        assertEquals("get_weather", callMessage.getToolCalls().get(0).name());
        // round 2: the answer can only be schema JSON
        ChatResponse r2 =
                model.call(
                        new Prompt(
                                List.of(
                                        user,
                                        callMessage,
                                        ToolResponseMessage.builder()
                                                .responses(
                                                        List.of(
                                                                new ToolResponseMessage
                                                                        .ToolResponse(
                                                                        callMessage
                                                                                .getToolCalls()
                                                                                .get(0)
                                                                                .id(),
                                                                        "get_weather",
                                                                        "18C, sunny")))
                                                .build()),
                                options));
        String text = r2.getResult().getOutput().getText();
        Map<String, Object> json = parseJson(text);
        assertTrue(String.valueOf(json.get("city")).contains("Munich"), "grounded: " + text);
        assertEquals(18.0, ((Number) json.get("temperature_c")).doubleValue(), 0.01, text);
    }

    @Test
    void chatClientEntityDogfood() {
        record Capital(String city, String country) {}
        Capital capital =
                ChatClient.create(model)
                        .prompt("What is the capital of France?")
                        .call()
                        .entity(
                                Capital.class,
                                ChatClient.EntityParamSpec::useProviderStructuredOutput);
        assertNotNull(capital);
        assertEquals("Paris", capital.city());
        assertTrue(!capital.country().isBlank());
    }
}
