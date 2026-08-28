package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import java.time.LocalDate;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.converter.BeanOutputConverter;

/**
 * A {@code java.time} field reaches the model as a SHAPE, not a bare string. Spring AI's schema
 * generator maps {@link LocalDate} to {@code {"type":"string","format":"date"}}, and until the
 * grammar compiler honoured {@code format} that field was constrained no more tightly than any
 * other string - the model could answer "next Tuesday" and satisfy the schema.
 *
 * <p>The first test pins the generator's half of the contract without loading a model, so a Spring
 * AI upgrade that stops emitting {@code format} is caught here rather than by a puzzling extraction
 * failure; the second is the end-to-end proof.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.spring.ai.DateFormatSchemaIT#modelAvailable")
class DateFormatSchemaIT {

    static final String REF = "hf.co/LiquidAI/LFM2.5-2.6B-GGUF/LFM2.5-2.6B-Q8_0.gguf";

    record Launch(String mission, LocalDate launchedOn) {}

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    private static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        .options(
                                JinferChatOptions.builder().maxTokens(256).temperature(0.0).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void springGeneratesADateFormatForJavaTime() {
        String schema = new BeanOutputConverter<>(Launch.class).getJsonSchema();
        assertTrue(schema.contains("\"format\""), schema);
        assertTrue(schema.replaceAll("\\s+", "").contains("\"format\":\"date\""), schema);
    }

    @Test
    void theModelCanOnlyAnswerWithARealDate() {
        Launch launch =
                ChatClient.create(model)
                        .prompt(
                                "Apollo 11 launched on 16 July 1969. Extract the mission name and"
                                        + " its launch date.")
                        .call()
                        .entity(
                                Launch.class,
                                ChatClient.EntityParamSpec::useProviderStructuredOutput);

        assertNotNull(launch);
        assertTrue(launch.mission().toLowerCase().contains("apollo"), launch.mission());
        // What format guarantees is the SHAPE: a value Jackson can read as a LocalDate. Without the
        // rule this field is any string and the model answers "16 July 1969" - measured, and the
        // deserializer throws on it. The VALUE tracks the model, as it does for every constrained
        // field (a 2.6B has answered 1677-11-19 here), so asserting the date itself would be
        // testing the checkpoint, not the compiler.
        assertNotNull(launch.launchedOn());
    }
}
