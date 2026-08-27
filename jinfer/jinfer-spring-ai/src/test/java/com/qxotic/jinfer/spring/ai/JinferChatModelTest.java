package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import org.junit.jupiter.api.Test;
import java.lang.reflect.Method;

/** Builder validation, no model needed: errors fail fast, before any GGUF is touched. */
class JinferChatModelTest {

    @Test
    void cacheSettingsRejectBeforeLoadingTheModel() {
        assertThrows(
                IllegalArgumentException.class, () -> JinferChatModel.builder().retainSessions(-1));
        IllegalArgumentException missing =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferChatModel.builder()
                                        .modelPath(Path.of("/also-missing.gguf"))
                                        .promptCache(Path.of("/missing-prompt-cache.jkv"))
                                        .build());
        assertTrue(missing.getMessage().contains("prompt cache does not exist"));
        assertThrows(NullPointerException.class, () -> JinferChatModel.builder().promptCache(null));
    }

    @Test
    void contextLengthHasOneSentinelAtEveryBuilder() {
        JinferChatModel.builder().contextLength(0);
        JinferEmbeddingModel.builder().contextLength(0);
        JinferDocumentPostProcessor.builder().contextLength(0);
        assertThrows(
                IllegalArgumentException.class, () -> JinferChatModel.builder().contextLength(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferEmbeddingModel.builder().contextLength(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferDocumentPostProcessor.builder().contextLength(-1));
    }

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
    void companionSourcesHaveTheSameExplicitBoundary() {
        JinferChatModel.builder()
                .companion("media", "hf.co/owner/repo/mmproj-F16.gguf")
                .companionPath("speculation", Path.of("models/mtp.gguf"));

        IllegalArgumentException path =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> JinferChatModel.builder().companion("media", "models/mmproj.gguf"));
        assertTrue(path.getMessage().contains("companionPath"));

        IllegalArgumentException url =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferChatModel.builder()
                                        .companion("media", "https://example.org/mmproj.gguf"));
        assertTrue(url.getMessage().contains("companionPath"));
        assertTrue(url.getMessage().contains("download"));
    }

    @Test
    void builderHasOneGenerationOptionsDoor() {
        assertThrows(NullPointerException.class, () -> JinferChatModel.builder().options(null));
        var methods =
                Arrays.stream(JinferChatModel.Builder.class.getDeclaredMethods())
                        .map(Method::getName)
                        .toList();
        assertTrue(methods.contains("options"));
        for (String removed :
                new String[] {
                    "defaultOptions",
                    "temperature",
                    "topP",
                    "topK",
                    "minP",
                    "maxTokens",
                    "seed",
                    "thinking",
                    "timeout"
                }) {
            assertFalse(
                    methods.contains(removed), removed + " is a second generation-options door");
        }
    }

    @Test
    void configuredOptionsOverrideModelRecommendationsFieldByField() {
        var recommended = new LoadedModel.SamplingDefaults(0.8f, 0.95f, 40, 0.05f);
        var configured =
                JinferChatOptions.builder()
                        .temperature(0.0)
                        .topK(1)
                        .seed(7L)
                        .thinking(false)
                        .timeout(Duration.ofSeconds(2))
                        .build();

        JinferChatOptions resolved =
                JinferChatModel.resolveDefaults("model.gguf", recommended, configured);

        assertEquals("model.gguf", resolved.getModel());
        assertEquals(0.0, resolved.getTemperature());
        assertEquals(0.95, resolved.getTopP(), 0.000001);
        assertEquals(1, resolved.getTopK());
        assertEquals(0.05, resolved.getMinP(), 0.000001);
        assertEquals(7L, resolved.getSeed());
        assertEquals(Boolean.FALSE, resolved.getThinking());
        assertEquals(Duration.ofSeconds(2), resolved.getTimeout());
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
