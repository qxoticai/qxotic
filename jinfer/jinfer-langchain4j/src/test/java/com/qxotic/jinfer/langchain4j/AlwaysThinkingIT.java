package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.ChatTemplate.ThinkingPolicy;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM2.5-8B-A1B has no non-thinking turn: thinking off is refused with the remedy, and the remedy,
 * a reasoning budget, is honoured with the span separated from the answer.
 */
@Tag("integration")
class AlwaysThinkingIT {

    private static Path always() {
        return TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0");
    }

    @Test
    void thinkingOffIsRefusedAtBuildWithTheRemedy() {
        UnsupportedFeatureException e =
                assertThrows(
                        UnsupportedFeatureException.class,
                        () ->
                                JinferChatModel.builder()
                                        .modelPath(always())
                                        .thinking(false)
                                        .build());
        assertTrue(e.getMessage().contains("always reasons"), e.getMessage());
        assertTrue(e.getMessage().contains("reasoning budget"), e.getMessage());
    }

    @Test
    void aBudgetCapsTheSpanAndTheAnswerStaysClean() {
        try (var m =
                JinferChatModel.builder()
                        .modelPath(always())
                        .temperature(0.0)
                        .maxOutputTokens(200)
                        .reasoningBudget(48)
                        .reasoningBudgetMessage("... Let me answer.")
                        .build()) {
            assertEquals(ThinkingPolicy.ALWAYS, m.thinkingPolicy());
            ChatResponse r =
                    m.chat(
                            ChatRequest.builder()
                                    .messages(
                                            SystemMessage.from(
                                                    "You are a terse assistant. Answer with the"
                                                            + " value only."),
                                            UserMessage.from("Capital of France?"))
                                    .build());
            assertTrue(r.aiMessage().text().contains("Paris"), r.aiMessage().text());
            assertTrue(
                    r.aiMessage().text().length() < 40,
                    "no leaked reasoning: " + r.aiMessage().text());
            assertNotNull(r.aiMessage().thinking(), "the span is separated");
            ChatResponse perRequest =
                    m.chat(
                            ChatRequest.builder()
                                    .messages(UserMessage.from("Capital of Italy? One word."))
                                    .parameters(
                                            JinferChatRequestParameters.builder()
                                                    .reasoningBudget(32)
                                                    .build())
                                    .build());
            assertTrue(
                    perRequest.aiMessage().text().contains("Rome"), perRequest.aiMessage().text());
        }
    }

    @Test
    void theSwitchableSiblingsKeepTheirPolicy() {
        try (var optional =
                JinferChatModel.builder()
                        .modelPath(TestModels.require("hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q8_0"))
                        .thinking(false)
                        .build()) {
            assertEquals(ThinkingPolicy.OPTIONAL, optional.thinkingPolicy());
            assertEquals(1, optional.chat("Capital of France? One word.").split("\\s+").length);
        }
        try (var instruct =
                JinferChatModel.builder()
                        .modelPath(TestModels.require("hf.co/LiquidAI/LFM2.5-350M-GGUF:Q8_0"))
                        .thinking(false)
                        .build()) {
            assertNotEquals(ThinkingPolicy.ALWAYS, instruct.thinkingPolicy());
        }
    }
}
