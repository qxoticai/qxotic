package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Config;
import com.qxotic.jinfer.Media;
import com.qxotic.jinfer.SpeechModel;
import com.qxotic.jinfer.SpeechOptions;
import com.qxotic.jinfer.SpeechState;
import dev.langchain4j.model.audio.TextToSpeechRequest;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.function.Predicate;
import org.junit.jupiter.api.Test;

/** The adapter's own behaviour, over a toy model: no GGUF, no weights, no kernels. */
final class JinferSpeechModelTest {

    @Test
    void aModelIsRequired() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> JinferSpeechModel.builder().build());
        assertEquals("exactly one of model(...) or modelPath(...) is required", e.getMessage());
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
    void synthesisReturnsWav() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            var audio = speech.synthesize("hello").audio();
            assertEquals("audio/wav", audio.mimeType());
            // 44-byte RIFF header + one 16-bit sample per PCM float
            assertEquals(44 + 2 * ToyModel.SAMPLES, audio.binaryData().length);
        }
    }

    @Test
    void aVoiceThisModelDoesNotHaveIsRefusedRatherThanIgnored() {
        try (var speech = JinferSpeechModel.builder().model(new ToyModel()).build()) {
            UnsupportedOperationException e =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    speech.synthesize(
                                            TextToSpeechRequest.builder("hi")
                                                    .voice("alloy")
                                                    .build()));
            assertTrue(e.getMessage().contains("alloy"), e.getMessage());
        }
    }

    @Test
    void anOversizedRequestIsRejectedBeforeAnySynthesis() {
        try (var speech =
                JinferSpeechModel.builder().model(new ToyModel()).maxInputChars(4).build()) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class, () -> speech.synthesize("far too long"));
            assertTrue(e.getMessage().contains("over the 4"), e.getMessage());
        }
    }

    @Test
    void closingTheAdapterClosesItsState() {
        ToyModel model = new ToyModel();
        var speech = JinferSpeechModel.builder().model(model).build();
        speech.close();
        assertTrue(model.state.closed, "the state the adapter minted is the adapter's to close");
        speech.close(); // idempotent
    }

    @Test
    void loadsARealModelThroughArchitectureDispatch() {
        Path gguf = com.qxotic.jinfer.testkit.ModelFixture.INFLECT_NANO_V2_Q8.require();
        try (var speech = JinferSpeechModel.builder().modelPath(gguf).build()) {
            var audio = speech.synthesize("Speech from a real model.").audio();
            assertEquals("audio/wav", audio.mimeType());
            assertTrue(audio.binaryData().length > 44 + 2 * 8000, "at least a third of a second");
        }
    }

    /** Emits one fixed clip per call and remembers the state it handed out. */
    private static final class ToyModel implements SpeechModel<Config, Void, ToyState> {

        static final int SAMPLES = 8;

        ToyState state;

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
            sink.test(new Media.Audio(new float[SAMPLES], 24000, 1));
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
