package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechState;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.List;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;
import org.springframework.ai.audio.tts.TextToSpeechOptions;
import org.springframework.ai.audio.tts.TextToSpeechPrompt;
import org.springframework.ai.audio.tts.TextToSpeechResponse;
import reactor.core.publisher.Flux;

/**
 * The adapter's own behaviour over a toy model - no GGUF, no weights, no kernels - plus one
 * fixture-gated pass over the real thing so the dispatch path is covered too.
 */
final class JinferSpeechModelTest {

    // ── builder ───────────────────────────────────────────────────────────

    @Test
    void aModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> JinferSpeechModel.builder().build());
        assertEquals("exactly one of model(...) or modelPath(...) is required", e.getMessage());
    }

    @Test
    void bothAModelAndAPathIsAmbiguous() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        JinferSpeechModel.builder()
                                .model(new ToyModel())
                                .modelPath(Path.of("/nonexistent.gguf"))
                                .build());
    }

    @Test
    void anArenaIsForTheLoadingPathOnly() {
        // a model the caller built already sits in an arena; a second one would own nothing
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferSpeechModel.builder()
                                        .model(new ToyModel())
                                        .arena(Arena.ofAuto())
                                        .build());
        assertTrue(e.getMessage().contains("arena"), e.getMessage());
    }

    // ── call ──────────────────────────────────────────────────────────────

    @Test
    void callReturnsWav() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            byte[] wav = speech.call("hello");
            // 44-byte RIFF header + one 16-bit sample per PCM float, over every clip
            assertEquals(44 + 2 * ToyModel.SAMPLES * ToyModel.CLIPS, wav.length);
            assertEquals("RIFF", new String(wav, 0, 4, java.nio.charset.StandardCharsets.US_ASCII));

            byte[] viaPrompt =
                    speech.call(new TextToSpeechPrompt("hello")).getResult().getOutput();
            assertEquals(wav.length, viaPrompt.length);
        }
    }

    @Test
    void speedIsTheOneRequestKnobThatSurvivesTranslation() {
        ToyModel model = new ToyModel();
        try (var speech = JinferSpeechModel.builder().model(model).speed(1.1).build()) {
            speech.call(new TextToSpeechPrompt("hello"));
            assertEquals(1.1, model.lastOptions.speed(), "the builder default when none is asked");

            speech.call(
                    new TextToSpeechPrompt(
                            "hello", TextToSpeechOptions.builder().speed(1.5).build()));
            assertEquals(1.5, model.lastOptions.speed(), "the request wins over the default");
        }
    }

    @Test
    void aRequestWithoutOptionsFallsBackToTheModelsOwnDefaults() {
        ToyModel model = new ToyModel();
        try (var speech = JinferSpeechModel.builder().model(model).build()) {
            speech.call(new TextToSpeechPrompt("hello"));
            assertNull(model.lastOptions.speed(), "nothing configured means nothing overridden");
        }
    }

    @Test
    void knobsThisInstanceDoesNotHaveAreRefusedRatherThanIgnored() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            assertTrue(rejected(speech, TextToSpeechOptions.builder().voice("nova")).contains("nova"));
            assertTrue(rejected(speech, TextToSpeechOptions.builder().format("mp3")).contains("mp3"));
            assertTrue(
                    rejected(speech, TextToSpeechOptions.builder().model("tts-1-hd"))
                            .contains("tts-1-hd"));
        }
    }

    private static String rejected(JinferSpeechModel speech, TextToSpeechOptions.Builder options) {
        return assertThrows(
                        UnsupportedOperationException.class,
                        () -> speech.call(new TextToSpeechPrompt("hello", options.build())))
                .getMessage();
    }

    @Test
    void anOversizedRequestIsRejectedBeforeAnySynthesis() {
        ToyModel model = new ToyModel();
        try (var speech =
                JinferSpeechModel.builder().model(model).maxInputChars(4).build()) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> speech.call(new TextToSpeechPrompt("far too long")));
            assertTrue(e.getMessage().contains("over the 4"), e.getMessage());
            assertEquals(0, model.clipsProduced, "nothing was synthesized");
        }
    }

    // ── stream ────────────────────────────────────────────────────────────

    @Test
    void streamEmitsOneElementPerClip() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            List<TextToSpeechResponse> clips =
                    speech.stream(new TextToSpeechPrompt("hello")).collectList().block();
            assertEquals(ToyModel.CLIPS, clips.size());
            // headerless PCM16 per clip: they concatenate, unlike a WAV per clip would
            assertEquals(2 * ToyModel.SAMPLES, clips.get(0).getResult().getOutput().length);
        }
    }

    @Test
    void cancellingTheSubscriptionCancelsTheSynthesis() {
        ToyModel model = new ToyModel();
        try (var speech = JinferSpeechModel.builder().model(model).build()) {
            Flux<TextToSpeechResponse> stream = speech.stream(new TextToSpeechPrompt("hello"));
            assertEquals(1, stream.take(1).collectList().block().size());
            assertTrue(
                    model.clipsProduced < ToyModel.CLIPS,
                    "a cancelled stream must stop synthesizing, produced " + model.clipsProduced);
        }
    }

    @Test
    void thePipelineIsFreedForTheNextRequest() {
        // the state is one serial pipeline held for the whole emission: a second stream must be
        // able to run after the first completes, not deadlock on a lock the first never released
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            speech.stream(new TextToSpeechPrompt("one")).collectList().block();
            assertEquals(
                    ToyModel.CLIPS,
                    speech.stream(new TextToSpeechPrompt("two")).collectList().block().size());
        }
    }

    // ── lifetime ──────────────────────────────────────────────────────────

    @Test
    void closingTheAdapterClosesItsState() {
        ToyModel model = new ToyModel();
        var speech = JinferSpeechModel.builder().model(model).build();
        speech.close();
        assertTrue(model.state.closed, "the state the adapter minted is the adapter's to close");
        speech.close(); // idempotent
    }

    // ── the real thing ────────────────────────────────────────────────────

    @Test
    void loadsARealModelThroughArchitectureDispatch() {
        Path gguf = ModelFixture.INFLECT_NANO_V2_Q8.require();
        try (var speech = JinferSpeechModel.builder().modelPath(gguf).build()) {
            byte[] wav = speech.call("Speech from a real model.");
            assertEquals("RIFF", new String(wav, 0, 4, java.nio.charset.StandardCharsets.US_ASCII));
            assertTrue(wav.length > 44 + 2 * 8000, "at least a third of a second: " + wav.length);
        }
    }

    // ── doubles ───────────────────────────────────────────────────────────

    /** Emits {@link #CLIPS} fixed clips, remembering what it was asked and how far it got. */
    private static final class ToyModel implements SpeechModel<Config, Void, ToyState> {

        static final int SAMPLES = 8;
        static final int CLIPS = 3;

        ToyState state;
        SpeechOptions lastOptions;
        int clipsProduced;

        @Override
        public Config config() {
            throw new UnsupportedOperationException();
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public ToyState newState(Arena arena, boolean adopt) {
            return state = new ToyState();
        }

        @Override
        public void speak(
                ToyState state, String text, SpeechOptions options, Predicate<Media.Audio> sink) {
            lastOptions = options;
            for (int i = 0; i < CLIPS; i++) {
                clipsProduced++;
                if (!sink.test(new Media.Audio(new float[SAMPLES], 24000, 1))) return;
            }
        }
    }

    private static final class ToyState implements SpeechState {
        boolean closed;

        @Override
        public void close() {
            closed = true;
        }
    }
}
