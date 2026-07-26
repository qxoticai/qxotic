package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The model-agnostic CONSTRAINED-GENERATION contract: whatever the family's reply format (span
 * markers, prompt-opened think, Harmony channels, no reasoning at all), a grammar or schema must
 * bind the OUTPUT text and nothing else. Wire syntax varies per family; these laws may not. The
 * documented output law is "optional boilerplate newlines + a string of the grammar's language"
 * (the post-close tolerance), so assertions strip before membership checks.
 *
 * <p>Parameterized by model: each concrete subclass names one GGUF (overridable via a system
 * property); the suite assume-skips when the file is absent.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
abstract class AbstractConstraintIT {

    /** The GGUF this subclass runs against. */
    abstract Path modelPath();

    JinferChatModel model;

    @BeforeAll
    void load() {
        Assumptions.assumeTrue(Files.exists(modelPath()), "model not found: " + modelPath());
        model =
                JinferChatModel.builder()
                        .modelPath(modelPath())
                        .contextLength(4096)
                        .maxOutputTokens(512)
                        .build();
    }

    private ChatResponse constrained(String question, String gbnf) {
        return model.chat(
                ChatRequest.builder()
                        .messages(UserMessage.from(question))
                        .parameters(JinferChatRequestParameters.builder().grammar(gbnf).build())
                        .build());
    }

    @Test
    void grammarAdmitsOnlyItsLanguage() {
        String text =
                constrained(
                                "Is the sky blue on a clear day? Answer strictly yes or no.",
                                "root ::= \"yes\" | \"no\"")
                        .aiMessage()
                        .text()
                        .strip();
        assertTrue(text.equals("yes") || text.equals("no"), "constrained output: '" + text + "'");
    }

    @Test
    void closedLabelClassification() {
        String text =
                constrained(
                                "Classify this support ticket: 'my parcel never arrived'. Answer"
                                        + " with the category only.",
                                "root ::= \"billing\" | \"shipping\" | \"warranty\" | \"other\"")
                        .aiMessage()
                        .text()
                        .strip();
        assertTrue(
                List.of("billing", "shipping", "warranty", "other").contains(text),
                "label: '" + text + "'");
    }

    @Test
    void jsonSchemaShapesTheReply() {
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the capital of France and roughly how"
                                                        + " many million people live in the"
                                                        + " city?"))
                                .responseFormat(
                                        ResponseFormat.builder()
                                                .type(ResponseFormatType.JSON)
                                                .jsonSchema(
                                                        JsonSchema.builder()
                                                                .rootElement(
                                                                        JsonObjectSchema.builder()
                                                                                .addStringProperty(
                                                                                        "city")
                                                                                .addNumberProperty(
                                                                                        "population_millions")
                                                                                .required(
                                                                                        "city",
                                                                                        "population_millions")
                                                                                .build())
                                                                .build())
                                                .build())
                                .build());
        Object parsed = JsonCodec.parse(r.aiMessage().text());
        assertTrue(parsed instanceof Map<?, ?>, r.aiMessage().text());
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertEquals(2, map.size(), "no extra keys: " + r.aiMessage().text());
        assertTrue(
                String.valueOf(map.get("city")).toLowerCase().contains("paris"),
                r.aiMessage().text());
        assertTrue(map.get("population_millions") instanceof Number, r.aiMessage().text());
    }

    @Test
    void reasoningStaysFreeUnderTheGrammar() {
        // channel scoping, family-agnostic: IF this family reasons, the grammar must not have
        // silenced it (assumption-gated - some families/vocabs have no reasoning at all)
        ChatResponse r =
                constrained(
                        "Is 1024 a power of two? Answer strictly yes or no.",
                        "root ::= \"yes\" | \"no\"");
        String text = r.aiMessage().text().strip();
        assertTrue(text.equals("yes") || text.equals("no"), text);
        Assumptions.assumeTrue(r.aiMessage().thinking() != null, "family does not reason");
        assertTrue(
                !r.aiMessage().thinking().isBlank(),
                "reasoning must flow unconstrained while output is grammar-bound");
    }

    @Test
    void grammarRejectsToolsLoudly() {
        ToolSpecification noop =
                ToolSpecification.builder()
                        .name("noop")
                        .parameters(JsonObjectSchema.builder().build())
                        .build();
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= \"x\"")
                                                        .toolSpecifications(noop)
                                                        .build())
                                        .build()));
    }
}
