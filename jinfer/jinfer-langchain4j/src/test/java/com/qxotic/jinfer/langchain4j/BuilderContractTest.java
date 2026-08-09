package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.chat.request.DefaultChatRequestParameters;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

/** What build() promises: rejection order, no leak on a rejected build, loaded-model knobs. */
final class BuilderContractTest {

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
    void aRejectedBuildFreesTheEngineItAlreadyBuilt() throws IOException {
        // a defaults modelName is checkable only against the LIVE engine, so this rejection
        // exercises the close-on-failure guard around the constructor tail. The freeing itself
        // is only indirectly observable (LeakWatch reports a stranded engine at GC); what this
        // asserts is the documented failure plus the shared weights staying serviceable.
        try (Arena weights = Arena.ofShared()) {
            LoadedModel<?> loaded = Models.load(ModelFixture.LFM25_350M_Q8.require(), weights);
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
        try (Arena weights = Arena.ofShared()) {
            LoadedModel<?> loaded = Models.load(ModelFixture.LFM25_350M_Q8.require(), weights);
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
        try (Arena weights = Arena.ofShared()) {
            LoadedModel<?> loaded = Models.load(ModelFixture.LFM25_350M_Q8.require(), weights);
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    JinferChatModel.builder()
                                            .model(loaded)
                                            .companion(
                                                    "media", ModelFixture.LFM25_350M_Q8.require())
                                            .build());
            assertEquals(
                    "companions are load-time settings; apply them when you build the LoadedModel"
                            + " passed to model(...)",
                    e.getMessage());
        }
    }
}
