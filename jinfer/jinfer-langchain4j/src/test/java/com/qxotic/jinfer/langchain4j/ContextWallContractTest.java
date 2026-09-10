package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.output.FinishReason;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * The context-overflow taxonomy, blocking lanes (the streaming wall pin lives in {@link
 * StreamingContractTest#hittingTheContextWallMidStreamKeepsThePartialsAndFinishesLength}). Three
 * distinct fates, three distinct signals: a prompt that cannot fit is refused BEFORE any state is
 * touched, with the counts and the remedy in the message; a generation that runs out of room ends
 * gracefully as LENGTH; prompt framing counts toward the context limit too.
 */
class ContextWallContractTest {

    private JinferChatModel model;

    @BeforeEach
    void load() {
        model =
                ChatFixtures.builder()
                        .contextLength(64) // small enough that walls are a few dozen tokens away
                        .temperature(0.0)
                        .build();
    }

    @AfterEach
    void close() {
        if (model != null) model.close();
    }

    @Test
    void promptOverCapacityIsRefusedWithTheCountsAndTheRemedy() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(
                                                        UserMessage.from("elaborate ".repeat(500)))
                                                .maxOutputTokens(8)
                                                .build()));
        assertTrue(e.getMessage().contains("context capacity"), e.getMessage());
        assertTrue(e.getMessage().contains("64 available"), e.getMessage());
        assertTrue(e.getMessage().contains("raise the context capacity"), e.getMessage());
    }

    @Test
    void promptFramingCountsTowardCapacity() {
        // The text alone fits; the template's final token pushes it past the wall.
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.chat(
                                        ChatRequest.builder()
                                                .messages(UserMessage.from("a".repeat(64)))
                                                .maxOutputTokens(8)
                                                .build()));
        assertTrue(e.getMessage().contains("context capacity"), e.getMessage());
    }

    @Test
    void generationHittingTheWallFinishesLengthNeverErrors() {
        // The model never emits a stop token, so generation must reach the context wall.
        ChatResponse r =
                model.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Count from 1 to 500, separated by commas."))
                                .maxOutputTokens(2000) // the knob allows more than the wall does
                                .build());
        assertEquals(FinishReason.LENGTH, r.finishReason());
        String text = r.aiMessage().text();
        assertTrue(!text.isEmpty() && text.chars().allMatch(c -> c == 'x'), text);
        assertTrue(
                r.tokenUsage().inputTokenCount() + r.tokenUsage().outputTokenCount() <= 64,
                "the wall is the context: " + r.tokenUsage());
    }
}
