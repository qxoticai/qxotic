package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.llm.Sampling;
import java.util.Map;
import org.junit.jupiter.api.Test;

class GenerationTest {

    @Test
    void anOmittedSeedStaysUnseeded() {
        Sampling defaults = new Sampling(0.8f, 0.95f, 40, 0.05f, null);
        assertNull(Generation.sampling(Map.of(), defaults).seed());
        assertEquals(42L, Generation.sampling(Map.of("seed", 42), defaults).seed());
    }

    @Test
    void reasoningKnobsAreNullUnlessGiven() {
        assertNull(Generation.reasoningMax(Map.of()));
        assertNull(Generation.reasoningMessage(Map.of()));
        assertEquals(64, Generation.reasoningMax(Map.of("reasoning_max_tokens", 64)));
        assertEquals(
                "... Let me wrap up.",
                Generation.reasoningMessage(Map.of("reasoning_message", "... Let me wrap up.")));
    }

    @Test
    void malformedEchoedArgumentsAreTheClientsFault() {
        // an assistant turn echoed with a model's broken JSON: a 400 through the worker's
        // IllegalArgumentException mapping, never a 500 with a stack trace
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Generation.parseArguments("{\"city\": Paris"));
        assertTrue(e.getMessage().contains("arguments"), e.getMessage());
    }

    @Test
    void reasoningEffortIsTheOpenAiSpellingOfThinking() {
        assertEquals(false, Generation.thinking(Map.of("reasoning_effort", "none"), true));
        assertEquals(true, Generation.thinking(Map.of("reasoning_effort", "low"), false));
        assertEquals(
                false,
                Generation.thinking(
                        Map.of(
                                "reasoning_effort",
                                "high",
                                "chat_template_kwargs",
                                Map.of("enable_thinking", false)),
                        true),
                "the explicit kwarg wins");
        assertEquals(true, Generation.thinking(Map.of(), true));
    }
}
