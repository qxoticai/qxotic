package com.qxotic.jinfer.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.audio.AudioTranscriptionRequest;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Optional;
import org.junit.jupiter.api.Test;

class JinferTranscriptionModelTest {

    @Test
    void requiresAModel() {
        assertThatThrownBy(() -> JinferTranscriptionModel.builder().build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("a model is required");
    }

    @Test
    void transcribesWavBytesThroughTheLangchain4jDoor() throws IOException {
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
            assertThat(transcriber.transcribe(wav).text()).isEqualTo(expected);
            assertThat(
                            transcriber.transcribeToText(
                                    Audio.builder().binaryData(wav).mimeType("audio/wav").build()))
                    .isEqualTo(expected);
            assertThatThrownBy(
                            () ->
                                    transcriber.transcribe(
                                            AudioTranscriptionRequest.builder()
                                                    .audio(
                                                            Audio.builder()
                                                                    .binaryData(wav)
                                                                    .mimeType("audio/wav")
                                                                    .build())
                                                    .language("fr")
                                                    .build()))
                    .isInstanceOf(UnsupportedFeatureException.class);
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
}
