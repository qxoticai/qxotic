package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import org.junit.jupiter.api.Test;

/**
 * A knob out of its range fails where it is set, with the range in the message, not at the first
 * request.
 */
class BuilderRangesTest {

    private static void refused(String part, Runnable call) {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, call::run);
        assertTrue(e.getMessage().contains(part), e.getMessage());
    }

    @Test
    void chatBuilderRefusesOutOfRangeSampling() {
        refused(">= 0", () -> JinferChatModel.builder().temperature(-1.0));
        refused("(0, 1]", () -> JinferChatModel.builder().topP(1.5));
        refused("(0, 1]", () -> JinferChatModel.builder().topP(0.0));
        refused(">= 0", () -> JinferChatModel.builder().topK(-1));
        refused("[0, 1]", () -> JinferChatModel.builder().minP(2.0));
        refused("-1", () -> JinferChatModel.builder().maxOutputTokens(-5));
        refused(">= 0", () -> JinferChatModel.builder().timeout(Duration.ofSeconds(-1)));
        refused(">= 0", () -> JinferChatModel.builder().contextLength(-1));
        JinferChatModel.builder()
                .temperature(0.0)
                .topP(1.0)
                .topK(0)
                .minP(0.0)
                .maxOutputTokens(-1)
                .timeout(Duration.ZERO);
    }

    @Test
    void speechBuilderRefusesANonPositiveSpeed() {
        refused("> 0", () -> JinferSpeechModel.builder().speed(0));
        refused("> 0", () -> JinferSpeechModel.builder().speed(Double.NaN));
        JinferSpeechModel.builder().speed(1.5);
    }
}
