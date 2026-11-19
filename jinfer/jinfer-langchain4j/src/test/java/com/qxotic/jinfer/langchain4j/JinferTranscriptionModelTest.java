package com.qxotic.jinfer.langchain4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.audio.Audio;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.audio.AudioTranscriptionRequest;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

class JinferTranscriptionModelTest {

    @Test
    void requiresAModel() {
        assertThatThrownBy(() -> JinferTranscriptionModel.builder().build())
                .isInstanceOf(IllegalArgumentException.class)
                .hasMessageContaining("a model is required");
    }

    @Test
    @Tag("integration")
    void transcribesWavBytesThroughTheLangchain4jDoor() throws IOException {
        // the JFK sentence, a public clip (make test-fixtures), and what it says (golden.tsv)
        byte[] wav = Files.readAllBytes(TestModels.fixture("parakeet/en.wav"));
        String expected =
                "Ask not what your country can do for you, ask what you can do for your country.";
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");

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
}
