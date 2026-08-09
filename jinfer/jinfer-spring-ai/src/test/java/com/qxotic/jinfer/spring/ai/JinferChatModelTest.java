package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** Builder validation, no model needed: errors fail fast, before any GGUF is touched. */
class JinferChatModelTest {

    @Test
    void aModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> JinferChatModel.builder().build());
        assertEquals(
                "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"), modelPath(...) or"
                        + " model(LoadedModel)",
                e.getMessage());
    }

    @Test
    void defaultOptionsAndKnobsAreMutuallyExclusive() {
        // the path need not exist: the conflict rejects BEFORE the GGUF is loaded
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferChatModel.builder()
                                        .modelPath(Path.of("/nonexistent.gguf"))
                                        .defaultOptions(JinferChatOptions.builder().build())
                                        .temperature(0.5)
                                        .build());
        assertTrue(e.getMessage().contains("mutually exclusive"), e.getMessage());
    }

    @Test
    void embeddingModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> JinferEmbeddingModel.builder().build());
        // the message teaches ALL THREE doors, ref form first
        assertEquals(
                "a model is required: model(\"hf.co/owner/repo:Q4_K_M\"), modelPath(...) or"
                        + " model(LoadedEmbedder)",
                e.getMessage());
    }
}
