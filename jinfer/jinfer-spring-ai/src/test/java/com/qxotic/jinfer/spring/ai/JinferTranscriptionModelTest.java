package com.qxotic.jinfer.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import java.io.IOException;
import java.lang.foreign.MemorySegment;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.audio.transcription.AudioTranscriptionOptions;
import org.springframework.ai.audio.transcription.AudioTranscriptionPrompt;
import org.springframework.core.io.ByteArrayResource;

/**
 * The adapter's own behaviour over a toy model - no GGUF, no weights, no kernels - plus one
 * fixture-gated pass over the real thing so the dispatch path is covered too.
 */
class JinferTranscriptionModelTest {

    @Test
    void requiresAModel() {
        assertThatThrownBy(() -> JinferTranscriptionModel.builder().build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("a model is required");
    }

    @Test
    void theLastModelSetterWins() {
        // the setters clear one another: the toy model set LAST wins over the bogus path, which
        // is therefore never opened - a successful build IS the assertion
        JinferTranscriptionModel.builder()
                .modelPath(Path.of("/nonexistent.gguf"))
                .model(new ToyModel())
                .build()
                .close();
    }

    @Test
    void callTranscribesTheResource() {
        try (var transcriber = JinferTranscriptionModel.builder().model(new ToyModel()).build()) {
            byte[] wav = AudioCodec.wav(new Media.Audio(new float[16_000], 16_000, 1));
            String text =
                    transcriber
                            .call(new AudioTranscriptionPrompt(new ByteArrayResource(wav)))
                            .getResult()
                            .getOutput();
            assertThat(text).isEqualTo("hello world");
        }
    }

    @Test
    void aRequestedModelIsRefusedRatherThanIgnored() {
        try (var transcriber = JinferTranscriptionModel.builder().model(new ToyModel()).build()) {
            byte[] wav = AudioCodec.wav(new Media.Audio(new float[16_000], 16_000, 1));
            AudioTranscriptionOptions options = () -> "whisper-1";
            assertThatThrownBy(
                            () ->
                                    transcriber.call(
                                            new AudioTranscriptionPrompt(
                                                    new ByteArrayResource(wav), options)))
                    .isInstanceOf(UnsupportedOperationException.class)
                    .hasMessageContaining("whisper-1")
                    .hasMessageContaining("bound to the loaded GGUF");
        }
    }

    @Test
    void audioAtTheWrongRateIsRefusedRatherThanDegraded() {
        try (var transcriber = JinferTranscriptionModel.builder().model(new ToyModel()).build()) {
            assertThatThrownBy(
                            () ->
                                    transcriber.transcribe(
                                            new Media.Audio(new float[8_000], 8_000, 1)))
                    .isInstanceOf(IllegalArgumentException.class)
                    .hasMessageContaining("8000 Hz")
                    .hasMessageContaining("16000 Hz");
        }
    }

    @Test
    void theDefaultStreamDoorRefuses() {
        try (var transcriber = JinferTranscriptionModel.builder().model(new ToyModel()).build()) {
            byte[] wav = AudioCodec.wav(new Media.Audio(new float[16_000], 16_000, 1));
            assertThatThrownBy(
                            () ->
                                    transcriber.stream(
                                                    new AudioTranscriptionPrompt(
                                                            new ByteArrayResource(wav)))
                                            .blockLast())
                    .isInstanceOf(UnsupportedOperationException.class);
        }
    }

    @Test
    void aRequestAfterCloseFailsInsteadOfReadingFreedMemory() {
        var transcriber = JinferTranscriptionModel.builder().model(new ToyModel()).build();
        transcriber.close();
        transcriber.close(); // idempotent
        assertThatThrownBy(() -> transcriber.transcribe(new Media.Audio(new float[16], 16_000, 1)))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }

    // ── the real thing ────────────────────────────────────────────────────

    @Test
    @Tag("integration")
    void transcribesWavBytesThroughTheSpringAiDoor() throws IOException {
        Optional<Path> fixturePath = fixture("tdt-0.6b-v3-q8_0-jfk.fixture.gguf");
        assumeTrue(fixturePath.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");

        byte[] wav;
        String expected;
        try (FileChannel channel = FileChannel.open(fixturePath.get(), StandardOpenOption.READ)) {
            GGUF fixture = GGUF.read(fixturePath.get());
            wav = AudioCodec.wav(new Media.Audio(floats(channel, fixture, "pcm"), 16_000, 1));
            expected = fixture.getValue(String.class, "fixture.transcript");
        }

        try (JinferTranscriptionModel transcriber =
                JinferTranscriptionModel.builder().modelPath(model).build()) {
            assertThat(
                            transcriber
                                    .call(new AudioTranscriptionPrompt(new ByteArrayResource(wav)))
                                    .getResult()
                                    .getOutput())
                    .isEqualTo(expected);
            Transcription typed = transcriber.transcribe(wav);
            assertThat(typed.text()).isEqualTo(expected);
            assertThat(typed.words()).isNotEmpty();
        }
    }

    private static float[] floats(FileChannel channel, GGUF gguf, String name) throws IOException {
        TensorEntry tensor =
                gguf.getTensors().stream()
                        .filter(entry -> entry.name().equals(name))
                        .findFirst()
                        .orElseThrow(() -> new IllegalStateException("fixture misses " + name));
        ByteBuffer bytes =
                ByteBuffer.allocate(Math.toIntExact(tensor.byteSize()))
                        .order(ByteOrder.LITTLE_ENDIAN);
        channel.read(bytes, gguf.getTensorDataOffset() + tensor.offset());
        bytes.flip();
        float[] values = new float[Math.toIntExact(tensor.totalNumberOfElements())];
        bytes.asFloatBuffer().get(values);
        return values;
    }

    private static Optional<Path> fixture(String name) {
        for (Path directory = Path.of("").toAbsolutePath();
                directory != null;
                directory = directory.getParent()) {
            Path candidate = directory.resolve("test-fixtures").resolve("parakeet").resolve(name);
            if (Files.isRegularFile(candidate)) return Optional.of(candidate);
        }
        return Optional.empty();
    }

    // ── doubles ───────────────────────────────────────────────────────────

    private static final class ToyModel
            implements com.qxotic.jinfer.TranscriptionModel<Void, Void, ToyState> {

        @Override
        public Void configuration() {
            return null;
        }

        @Override
        public Void weights() {
            return null;
        }

        @Override
        public int sampleRate() {
            return 16_000;
        }

        @Override
        public ToyState newState() {
            return new ToyState();
        }

        @Override
        public ToyState newState(MemoryArena<MemorySegment> arena) {
            return newState();
        }

        @Override
        public Transcription transcribe(ToyState state, float[] pcm) {
            return new Transcription(
                    "hello world",
                    List.of(
                            new Transcription.Token(" hello", 0.0, 0.4, 0.9),
                            new Transcription.Token(" world", 0.5, 0.9, 0.8)));
        }
    }

    private static final class ToyState extends RuntimeState {
        @Override
        protected void releaseResources() {}
    }
}
