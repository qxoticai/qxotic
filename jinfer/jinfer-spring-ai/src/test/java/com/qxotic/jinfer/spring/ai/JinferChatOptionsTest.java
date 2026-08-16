package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;

import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.prompt.ChatOptions;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

/** The option merge matrix used by Spring's ChatClient and direct ChatModel calls. */
class JinferChatOptionsTest {

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
                .topK(32)
                .minP(0.1)
                .maxTokens(128)
                .stopSequences(List.of("STOP"))
                .toolCallbacks(List.of(noopTool()))
                .toolContext(Map.of("k", "v"))
                .seed(7L)
                .thinking(false)
                .timeout(Duration.ofSeconds(3))
                .outputSchema("{\"type\":\"object\"}")
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
        assertEquals(32, copy.getTopK());
        assertEquals(0.1, copy.getMinP());
        assertEquals(128, copy.getMaxTokens());
        assertEquals(List.of("STOP"), copy.getStopSequences());
        assertEquals(1, copy.getToolCallbacks().size());
        assertEquals(Map.of("k", "v"), copy.getToolContext());
        assertEquals(7L, copy.getSeed());
        assertEquals(Boolean.FALSE, copy.getThinking());
        assertEquals(Duration.ofSeconds(3), copy.getTimeout());
        assertEquals("{\"type\":\"object\"}", copy.getOutputSchema());
    }

    @Test
    void foreignOptionsAreAdaptedWithoutModelDefaults() {
        ChatOptions foreign = ChatOptions.builder().temperature(0.1).maxTokens(5).build();
        JinferChatOptions adapted = JinferChatOptions.from(foreign);
        assertEquals(0.1, adapted.getTemperature());
        assertEquals(5, adapted.getMaxTokens());
        assertNull(adapted.getTopP());
        assertNull(adapted.getSeed());
        assertNull(adapted.getThinking());
    }

    @Test
    void jinferOptionsNeedNoAdaptation() {
        JinferChatOptions options = fullyPopulated();
        assertSame(options, JinferChatOptions.from(options));
    }

    @Test
    void foreignToolOptionsAreAdapted() {
        ToolCallingChatOptions foreign =
                ToolCallingChatOptions.builder()
                        .toolCallbacks(List.of(noopTool()))
                        .toolContext(Map.of("x", 1))
                        .build();
        JinferChatOptions adapted = JinferChatOptions.from(foreign);
        assertEquals(1, adapted.getToolCallbacks().size());
        assertEquals(Map.of("x", 1), adapted.getToolContext());
    }

    @Test
    void requestOptionsOverrideDefaultsAndInheritTheRest() {
        JinferChatOptions defaults =
                JinferChatOptions.builder()
                        .temperature(0.7)
                        .seed(1L)
                        .minP(0.1)
                        .thinking(true)
                        .timeout(Duration.ofSeconds(30))
                        .outputSchema("old")
                        .build();
        JinferChatOptions request =
                JinferChatOptions.builder()
                        .topP(0.8)
                        .seed(2L)
                        .minP(0.0)
                        .thinking(false)
                        .timeout(Duration.ofSeconds(2))
                        .outputSchema("new")
                        .build();

        Prompt effective =
                JinferChatModel.effectivePrompt(
                        new Prompt(new UserMessage("hello"), request), defaults);
        JinferChatOptions merged = (JinferChatOptions) effective.getOptions();

        assertEquals(List.of(new UserMessage("hello")), effective.getInstructions());
        assertEquals(0.7, merged.getTemperature(), "unset request fields inherit defaults");
        assertEquals(0.8, merged.getTopP());
        assertEquals(2L, merged.getSeed());
        assertEquals(0.0, merged.getMinP());
        assertEquals(Boolean.FALSE, merged.getThinking());
        assertEquals(Duration.ofSeconds(2), merged.getTimeout());
        assertEquals("new", merged.getOutputSchema());
    }

    @Test
    void portableRequestOptionsKeepJinferDefaults() {
        JinferChatOptions defaults = fullyPopulated();
        ChatOptions request = ChatOptions.builder().temperature(0.5).build();

        Prompt effective =
                JinferChatModel.effectivePrompt(
                        new Prompt(new UserMessage("hello"), request), defaults);
        JinferChatOptions merged = (JinferChatOptions) effective.getOptions();

        assertEquals(0.5, merged.getTemperature());
        assertEquals(7L, merged.getSeed());
        assertEquals(0.1, merged.getMinP());
        assertEquals(Boolean.FALSE, merged.getThinking());
        assertEquals("{\"type\":\"object\"}", merged.getOutputSchema());
    }

    @Test
    void requestToolsUseSpringCompositionExactlyOnce() {
        ToolCallback first = noopTool();
        ToolCallback second = noopTool();
        JinferChatOptions defaults =
                JinferChatOptions.builder()
                        .toolCallbacks(List.of(first))
                        .toolContext(Map.of("shared", "default", "left", 1))
                        .build();
        JinferChatOptions request =
                JinferChatOptions.builder()
                        .toolCallbacks(List.of(second))
                        .toolContext(Map.of("shared", "request", "right", 2))
                        .build();

        Prompt effective =
                JinferChatModel.effectivePrompt(
                        new Prompt(new UserMessage("hello"), request), defaults);
        JinferChatOptions merged = (JinferChatOptions) effective.getOptions();

        assertEquals(List.of(first, second), merged.getToolCallbacks());
        assertEquals(Map.of("shared", "request", "left", 1, "right", 2), merged.getToolContext());
    }

    @Test
    void requestWithoutOptionsReusesDefaults() {
        JinferChatOptions defaults = fullyPopulated();
        Prompt requested = new Prompt(new UserMessage("hello"));

        Prompt effective = JinferChatModel.effectivePrompt(requested, defaults);

        assertEquals(requested.getInstructions(), effective.getInstructions());
        assertSame(defaults, effective.getOptions());
    }
}
