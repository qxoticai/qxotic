package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/** Options plumbing: mutate/copy semantics and the foreign-options tolerance the advisor needs. */
class JinferChatOptionsTest {

    @Test
    void minPCopiesOntoAndMutates() {
        JinferChatOptions base = JinferChatOptions.builder().minP(0.1).build();
        assertEquals(0.1, base.mutate().build().getMinP());
        JinferChatOptions override = JinferChatOptions.builder().minP(0.0).build();
        assertEquals(0.0, JinferChatOptions.copyOnto(base, override).getMinP());
        // a foreign sparse options object must not wipe it
        ChatOptions foreign = ChatOptions.builder().temperature(0.5).build();
        assertEquals(0.1, JinferChatOptions.copyOnto(base, foreign).getMinP());
    }

    private static ToolCallback noopTool() {
        ToolDefinition def = DefaultToolDefinition.builder().name("noop").inputSchema("{}").build();
        return new ToolCallback() {
            @Override
            public ToolDefinition getToolDefinition() {
                return def;
            }

            @Override
            public String call(String toolInput) {
                return "";
            }
        };
    }

    private static JinferChatOptions fullyPopulated() {
        return JinferChatOptions.builder()
                .model("m.gguf")
                .temperature(0.7)
                .topP(0.9)
                .maxTokens(128)
                .stopSequences(List.of("STOP"))
                .toolCallbacks(List.of(noopTool()))
                .toolContext(Map.of("k", "v"))
                .seed(7L)
                .thinking(false)
                .timeout(Duration.ofSeconds(3))
                .build();
    }

    @Test
    void builderDefaultsAreNull() {
        JinferChatOptions o = JinferChatOptions.builder().build();
        assertNull(o.getModel());
        assertNull(o.getTemperature());
        assertNull(o.getMaxTokens());
        assertNull(o.getSeed());
        assertNull(o.getThinking());
        assertNull(o.getTimeout());
    }

    @Test
    void mutateRoundTripsEveryField() {
        JinferChatOptions copy = fullyPopulated().mutate().build();
        assertEquals("m.gguf", copy.getModel());
        assertEquals(0.7, copy.getTemperature());
        assertEquals(0.9, copy.getTopP());
        assertEquals(128, copy.getMaxTokens());
        assertEquals(List.of("STOP"), copy.getStopSequences());
        assertEquals(1, copy.getToolCallbacks().size());
        assertEquals(Map.of("k", "v"), copy.getToolContext());
        assertEquals(7L, copy.getSeed());
        assertEquals(Boolean.FALSE, copy.getThinking());
        assertEquals(Duration.ofSeconds(3), copy.getTimeout());
    }

    @Test
    void copyOntoFromForeignCarriesCommonFieldsOnly() {
        JinferChatOptions base = fullyPopulated();
        ChatOptions foreign = ChatOptions.builder().temperature(0.1).maxTokens(5).build();
        JinferChatOptions copied = JinferChatOptions.copyOnto(base, foreign);
        assertEquals(0.1, copied.getTemperature());
        assertEquals(5, copied.getMaxTokens());
        // merge semantics: foreign options are sparse, so their nulls must NOT wipe configured
        // defaults - only the fields they actually set override
        assertEquals(List.of("STOP"), copied.getStopSequences());
        assertEquals(0.9, copied.getTopP());
        assertEquals("m.gguf", copied.getModel());
        assertEquals(7L, copied.getSeed());
        assertEquals(Boolean.FALSE, copied.getThinking());
    }

    @Test
    void copyOntoFromEmptyForeignPreservesEverything() {
        JinferChatOptions base = fullyPopulated();
        JinferChatOptions copied = JinferChatOptions.copyOnto(base, ChatOptions.builder().build());
        assertEquals(0.7, copied.getTemperature());
        assertEquals(0.9, copied.getTopP());
        assertEquals(128, copied.getMaxTokens());
        assertEquals(List.of("STOP"), copied.getStopSequences());
        assertEquals("m.gguf", copied.getModel());
        assertEquals(7L, copied.getSeed());
    }

    @Test
    void copyOntoFromToolCallingOptionsCarriesTools() {
        ToolCallingChatOptions foreign =
                ToolCallingChatOptions.builder()
                        .toolCallbacks(List.of(noopTool()))
                        .toolContext(Map.of("x", 1))
                        .build();
        JinferChatOptions copied =
                JinferChatOptions.copyOnto(JinferChatOptions.builder().build(), foreign);
        assertEquals(1, copied.getToolCallbacks().size());
        assertEquals(Map.of("x", 1), copied.getToolContext());
    }

    @Test
    void copyOntoFromJinferCarriesExtras() {
        JinferChatOptions copied =
                JinferChatOptions.copyOnto(JinferChatOptions.builder().build(), fullyPopulated());
        assertEquals(7L, copied.getSeed());
        assertEquals(Boolean.FALSE, copied.getThinking());
        assertEquals(Duration.ofSeconds(3), copied.getTimeout());
    }

    @Test
    void outputSchemaRoundTrips() {
        JinferChatOptions o =
                JinferChatOptions.builder().outputSchema("{\"type\":\"object\"}").build();
        assertEquals("{\"type\":\"object\"}", o.getOutputSchema());
        assertEquals("{\"type\":\"object\"}", o.mutate().build().getOutputSchema());
    }

    @Test
    void copyOntoCarriesSchemaFromStructuredOptions() {
        JinferChatOptions source =
                JinferChatOptions.builder().outputSchema("{\"type\":\"object\"}").build();
        JinferChatOptions copied =
                JinferChatOptions.copyOnto(JinferChatOptions.builder().build(), source);
        assertEquals("{\"type\":\"object\"}", copied.getOutputSchema());
    }

    @Test
    void copyOntoFromPlainOptionsLeavesSchemaAlone() {
        JinferChatOptions base =
                JinferChatOptions.builder().outputSchema("{\"type\":\"object\"}").build();
        JinferChatOptions copied =
                JinferChatOptions.copyOnto(base, ChatOptions.builder().temperature(0.5).build());
        assertEquals("{\"type\":\"object\"}", copied.getOutputSchema());
    }
}
