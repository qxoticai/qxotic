package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.agent.tool.ToolSpecification;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The grammar/seed laws E2E on the smallest chat model (LFM2.5 350M): a GBNF gate admits only its
 * language; the same seed replays byte-identically at temperature 1.0 and different seeds diverge;
 * a standing grammar on the MODEL constrains plain requests (the framework-merge path AiServices
 * rides); the two conflict rejections are loud.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferRequestParametersIT {

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_350M_Q8.require())
                        .contextLength(2048)
                        .maxOutputTokens(48)
                        .build();
    }

    @Test
    void grammarAdmitsOnlyItsLanguage() {
        ChatRequest gated =
                ChatRequest.builder()
                        .messages(UserMessage.from("Is the sky blue? Answer strictly yes or no."))
                        .parameters(
                                JinferChatRequestParameters.builder()
                                        .grammar("root ::= \"yes\" | \"no\"")
                                        .build())
                        .build();
        String text = model.chat(gated).aiMessage().text();
        assertTrue(text.equals("yes") || text.equals("no"), "constrained output: '" + text + "'");
    }

    @Test
    void sameSeedReplaysByteIdentically() {
        ChatRequest req = creative(1234L);
        String a = model.chat(req).aiMessage().text();
        String b = model.chat(req).aiMessage().text();
        assertEquals(a, b, "same seed at temperature 1.0 must replay identically");
    }

    @Test
    void differentSeedsDiverge() {
        String a = model.chat(creative(1L)).aiMessage().text();
        String b = model.chat(creative(2L)).aiMessage().text();
        assertNotEquals(a, b, "different seeds at temperature 1.0 should diverge");
    }

    @Test
    void standingGrammarConstrainsPlainRequests() {
        // the AiServices path: jinfer defaults on the model, a request with NO jinfer parameters -
        // the framework merge (defaults.overrideWith(request)) must carry the grammar through
        try (JinferChatModel gate =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_350M_Q8.require())
                        .contextLength(2048)
                        .defaultRequestParameters(
                                JinferChatRequestParameters.builder()
                                        .grammar("root ::= \"positive\" | \"negative\"")
                                        .maxOutputTokens(8)
                                        .build())
                        .build()) {
            String text =
                    gate.chat(UserMessage.from("Sentiment of: 'this update is wonderful'"))
                            .aiMessage()
                            .text();
            assertTrue(
                    text.equals("positive") || text.equals("negative"),
                    "standing grammar must constrain plain requests: '" + text + "'");
        }
    }

    @Test
    void grammarConflictsAreLoud() {
        var grammarParams = JinferChatRequestParameters.builder().grammar("root ::= \"x\"").build();
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= \"x\"")
                                                        .toolSpecifications(
                                                                ToolSpecification.builder()
                                                                        .name("noop")
                                                                        .build())
                                                        .build())
                                        .build()),
                "grammar + tools");
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        model.chat(
                                ChatRequest.builder()
                                        .messages(UserMessage.from("hi"))
                                        .parameters(
                                                JinferChatRequestParameters.builder()
                                                        .grammar("root ::= \"x\"")
                                                        .responseFormat(ResponseFormat.JSON)
                                                        .build())
                                        .build()),
                "grammar + JSON response format");
        // and the params object alone is inert - conflicts reject at chat time, loudly
        assertEquals("root ::= \"x\"", grammarParams.grammar());
    }

    private static ChatRequest creative(long seed) {
        return ChatRequest.builder()
                .messages(UserMessage.from("Invent a name for a sailing boat. One name only."))
                .parameters(
                        JinferChatRequestParameters.builder()
                                .seed(seed)
                                .temperature(1.0)
                                .maxOutputTokens(24)
                                .build())
                .build();
    }
}
