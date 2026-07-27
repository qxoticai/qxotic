package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * Grammar-constrained output on a CHANNEL-FRAMED family (Harmony): the analysis channel must reason
 * freely while the final channel is grammar-bound - the capability the channel-aware constraint
 * exists for. Written FAILING against the token-id gate (which knows only think markers and
 * constrains Harmony's channel headers from token zero).
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class GptOssStructuredOutputIT {

    static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.GPTOSS_20B_Q8.require())
                        .contextLength(4096)
                        .maxOutputTokens(384)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Test
    void grammarConstrainsTheFinalChannelOnly() {
        String flatJson =
                "root ::= \"{\" ws \"\\\"city\\\"\" ws \":\" ws str ws \",\" ws"
                        + " \"\\\"population_millions\\\"\" ws \":\" ws num ws \"}\"\n"
                        + "str ::= \"\\\"\" [a-zA-Z ]+ \"\\\"\"\n"
                        + "num ::= [0-9] [0-9]? (\".\" [0-9])?\n"
                        + "ws ::= \" \"?";
        var response =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "What is the largest city in France and roughly"
                                                        + " how many million people live there?"
                                                        + " Answer as JSON with fields city and"
                                                        + " population_millions."))
                                .parameters(
                                        JinferChatRequestParameters.builder()
                                                .grammar(flatJson)
                                                .build())
                                .build());
        String text = response.aiMessage().text();
        Object parsed = JsonCodec.parse(text); // the grammar must make this a guarantee
        assertTrue(parsed instanceof Map<?, ?>, "final channel must be the JSON object: " + text);
        Map<?, ?> map = (Map<?, ?>) parsed;
        assertTrue(
                String.valueOf(map.get("city")).toLowerCase().contains("paris"),
                "grounded answer: " + text);
        assertTrue(map.get("population_millions") instanceof Number, text);
        // and the reasoning channel must have stayed FREE - Harmony always analyses first
        assertNotNull(response.aiMessage().thinking(), "analysis channel should be present");
        assertTrue(
                !response.aiMessage().thinking().isBlank(),
                "reasoning must flow unconstrained while the final channel is grammar-bound");
    }
}
