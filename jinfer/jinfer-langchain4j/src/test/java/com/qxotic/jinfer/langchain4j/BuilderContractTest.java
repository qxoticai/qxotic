package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** What build() promises: rejection order, no leak on a rejected build, loaded-model knobs. */
final class BuilderContractTest {

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
        JinferScoringModel.builder().contextLength(0);
        assertThrows(
                IllegalArgumentException.class, () -> JinferChatModel.builder().contextLength(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferEmbeddingModel.builder().contextLength(-1));
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferScoringModel.builder().contextLength(-1));
    }

    @Test
    void unsupportedDefaultsRejectBeforeTheWeightsEverMap() {
        // core merges defaults UNDER each request, so a request can add what defaults lack but
        // can never unset an unsupported knob - build-fatal, and checked before the load. The
        // nonexistent path is the proof of ordering: had the load run first, the failure would
        // be a missing-file error, not the documented rejection.
        assertThrows(
                UnsupportedFeatureException.class,
                () ->
                        JinferChatModel.builder()
                                .modelPath(Path.of("/nonexistent-model.gguf"))
                                .defaultRequestParameters(
                                        DefaultChatRequestParameters.builder()
                                                .frequencyPenalty(0.5)
                                                .build())
                                .build());
    }

    @Test
    void defaultParametersOverrideModelRecommendations() {
        var recommended = new LoadedModel.SamplingDefaults(0.8f, 0.95f, 40, 0.05f);
        var builder =
                JinferChatModel.builder()
                        .defaultRequestParameters(
                                JinferChatRequestParameters.builder()
                                        .temperature(0.6)
                                        .topP(0.9)
                                        .topK(20)
                                        .minP(0.1)
                                        .maxOutputTokens(128)
                                        .seed(3L)
                                        .build());

        var resolved = JinferChatModel.resolveDefaults("model.gguf", recommended, builder);

        assertEquals("model.gguf", resolved.modelName());
        assertEquals(0.6, resolved.temperature());
        assertEquals(0.9, resolved.topP());
        assertEquals(20, resolved.topK());
        assertEquals(0.1, resolved.minP());
        assertEquals(128, resolved.maxOutputTokens());
        assertEquals(3L, resolved.seed());
    }

    @Test
    void explicitBuilderSettersOverrideDefaultParameters() {
        var recommended = new LoadedModel.SamplingDefaults(0.8f, 0.95f, 40, 0.05f);
        var builder =
                JinferChatModel.builder()
                        .defaultRequestParameters(
                                JinferChatRequestParameters.builder()
                                        .temperature(0.6)
                                        .topP(0.9)
                                        .topK(20)
                                        .minP(0.1)
                                        .maxOutputTokens(128)
                                        .seed(3L)
                                        .build())
                        .temperature(0.0)
                        .topP(0.7)
                        .topK(1)
                        .minP(0.0)
                        .maxOutputTokens(16)
                        .seed(7L);

        var resolved = JinferChatModel.resolveDefaults("model.gguf", recommended, builder);

        assertEquals(0.0, resolved.temperature());
        assertEquals(0.7, resolved.topP());
        assertEquals(1, resolved.topK());
        assertEquals(0.0, resolved.minP());
        assertEquals(16, resolved.maxOutputTokens());
        assertEquals(7L, resolved.seed());
    }

    @Test
    void requestParametersStillOverrideEveryBuilderProvenance() {
        var defaults =
                JinferChatModel.resolveDefaults(
                        "model.gguf",
                        LoadedModel.SamplingDefaults.NONE,
                        JinferChatModel.builder()
                                .defaultRequestParameters(
                                        JinferChatRequestParameters.builder()
                                                .temperature(0.6)
                                                .seed(3L)
                                                .build())
                                .temperature(0.2)
                                .seed(7L));
        var request = JinferChatRequestParameters.builder().temperature(1.0).seed(11L).build();

        var resolved = defaults.overrideWith(request);

        assertEquals(1.0, resolved.temperature());
        assertEquals(11L, resolved.seed());
    }

    @Test
    void aRejectedBuildFreesTheEngineItAlreadyBuilt() throws IOException {
        // a defaults modelName is checkable only against the LIVE engine, so this rejection
        // exercises the close-on-failure guard around the constructor tail. The freeing itself
        // is only indirectly observable (LeakWatch reports a stranded engine at GC); what this
        // asserts is the documented failure plus the shared weights staying serviceable.
        try (Arena weights = Arenas.newCrossThread()) {
            LoadedModel<?> loaded =
                    Models.load(
                            TestModels.require(
                                    "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"),
                            weights);
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            JinferChatModel.builder()
                                    .model(loaded)
                                    .defaultRequestParameters(
                                            DefaultChatRequestParameters.builder()
                                                    .modelName("someone-else")
                                                    .build())
                                    .build());
            try (JinferChatModel model = JinferChatModel.builder().model(loaded).build()) {
                assertTrue(model.chat("Say hi.").length() > 0);
            }
        }
    }

    @Test
    void aLoadedModelStillTakesTheContextKnob() throws IOException {
        // contextLength sizes the STATE, which the engine allocates - it is not a load-time
        // setting, and LoadedModel has no knob to carry it. Refusing it here used to pin every
        // forked pipeline to the 4096 default with advice that could not be followed. Both
        // directions are pinned: a larger window is accepted, and the value demonstrably
        // reaches the state - a 64-token window refuses a prompt the default would take.
        try (Arena weights = Arenas.newCrossThread()) {
            LoadedModel<?> loaded =
                    Models.load(
                            TestModels.require(
                                    "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"),
                            weights);
            try (JinferChatModel model =
                    JinferChatModel.builder().model(loaded).contextLength(8192).build()) {
                assertTrue(model.chat("Say hi.").length() > 0);
            }
            try (JinferChatModel model =
                    JinferChatModel.builder().model(loaded).contextLength(64).build()) {
                IllegalArgumentException e =
                        assertThrows(
                                IllegalArgumentException.class,
                                () -> model.chat("lorem ipsum dolor sit amet ".repeat(60)));
                assertTrue(e.getMessage().contains("context capacity"), e.getMessage());
            }
        }
    }

    @Test
    void companionsStillBelongToTheLoad() throws IOException {
        try (Arena weights = Arenas.newCrossThread()) {
            LoadedModel<?> loaded =
                    Models.load(
                            TestModels.require(
                                    "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"),
                            weights);
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    JinferChatModel.builder()
                                            .model(loaded)
                                            .companion(
                                                    "media",
                                                    TestModels.require(
                                                            "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"))
                                            .build());
            assertEquals(
                    "companions are load-time settings; apply them when you build the LoadedModel"
                            + " passed to model(...)",
                    e.getMessage());
        }
    }
}
