package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.nio.file.Files;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * Structured output on a channel-framed model (Harmony): {@code outputSchema} must bind ONLY the
 * final channel while the analysis channel reasons freely - the langchain4j twin proves this via
 * raw GBNF; this is the spring surface's schema path. Model-gated (20B): assume-skips.
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
        Assumptions.assumeTrue(
                Files.exists(ModelFixture.GPTOSS_20B_Q8.path()),
                "model not found: " + ModelFixture.GPTOSS_20B_Q8.path());
        model =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.GPTOSS_20B_Q8.path())
                        .contextLength(4096)
                        .maxTokens(384)
                        .build();
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
        Object parsed = JsonCodec.parse(text); // the schema grammar must make this a guarantee
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
