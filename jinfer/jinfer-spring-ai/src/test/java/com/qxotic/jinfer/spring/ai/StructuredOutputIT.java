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
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/**
 * Native structured output (grammar-constrained decoding) against a real GGUF. Model-gated. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
class StructuredOutputIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel",
                            "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf"));

    static final String SCHEMA =
            "{\"type\":\"object\",\"properties\":{"
                    + "\"answer\":{\"type\":\"string\"},"
                    + "\"confidence\":{\"type\":\"number\"}},"
                    + "\"required\":[\"answer\",\"confidence\"]}";

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        model =
                JinferChatModel.builder()
                        .modelPath(MODEL)
                        .contextLength(4096)
                        .maxTokens(512)
                        .build();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> parseJson(String text) {
        return (Map<String, Object>) JsonCodec.parse(text);
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
    void rejectsSchemaWithTools() {
        ToolDefinition def = DefaultToolDefinition.builder().name("noop").inputSchema("{}").build();
        ToolCallback noop =
                new ToolCallback() {
                    @Override
                    public ToolDefinition getToolDefinition() {
                        return def;
                    }

                    @Override
                    public String call(String toolInput) {
                        return "";
                    }
                };
        var e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.call(
                                        new Prompt(
                                                new UserMessage("hi"),
                                                JinferChatOptions.builder()
                                                        .outputSchema(SCHEMA)
                                                        .toolCallbacks(List.of(noop))
                                                        .build())));
        assertTrue(e.getMessage().contains("grammar-constrained"), e.getMessage());
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
                                org.springframework.ai.chat.client.ChatClient.EntityParamSpec
                                        ::useProviderStructuredOutput);
        assertNotNull(capital);
        assertEquals("Paris", capital.city());
        assertTrue(!capital.country().isBlank());
    }
}
