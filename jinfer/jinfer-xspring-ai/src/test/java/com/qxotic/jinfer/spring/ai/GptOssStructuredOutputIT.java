package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Structured output on a channel-framed model (Harmony): {@code outputSchema} must bind ONLY the
 * final channel while the analysis channel reasons freely - the langchain4j twin proves this via
 * raw GBNF; this is the spring surface's schema path. Model-gated (20B) via {@link TestModels}.
 */
@Tag("integration")
class GptOssStructuredOutputIT {

    static final String SCHEMA =
            "{\"type\":\"object\",\"properties\":{"
                    + "\"city\":{\"type\":\"string\"},"
                    + "\"population_millions\":{\"type\":\"number\"}},"
                    + "\"required\":[\"city\",\"population_millions\"]}";

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/unsloth/gpt-oss-20b-GGUF/gpt-oss-20b-Q8_0.gguf"))
                        .contextLength(4096)
                        .options(JinferChatOptions.builder().maxTokens(384).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void schemaBindsTheFinalChannelWhileAnalysisReasonsFree() {
        ChatResponse r =
                model.call(
                        new Prompt(
                                new UserMessage(
                                        "What is the largest city in France and roughly how many"
                                                + " million people live there?"),
                                JinferChatOptions.builder().outputSchema(SCHEMA).build()));
        String text = r.getResult().getOutput().getText();
        Object parsed = Json.parse(text); // the schema grammar must make this a guarantee
        assertInstanceOf(Map.class, parsed, text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertTrue(String.valueOf(map.get("city")).toLowerCase().contains("paris"), text);
        assertInstanceOf(Number.class, map.get("population_millions"), text);
        // the reasoning channel stayed FREE - Harmony always analyses first
        Object thinking = r.getResult().getOutput().getMetadata().get("thinking");
        assertNotNull(thinking, "analysis channel should be present");
        assertTrue(
                !thinking.toString().isBlank(),
                "reasoning must flow unconstrained while the final channel is schema-bound");
    }
}
