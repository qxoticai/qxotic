package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/** Architecture dispatch and the {@link TranscriptionModel} contract over a real checkpoint. */
class ParakeetProviderTest {

    @Test
    void dispatchLoadsAndTranscribes() throws IOException {
        Optional<Path> fixturePath = Fixtures.fixture("tdt-0.6b-v3-f16-jfk.fixture.gguf");
        assumeTrue(fixturePath.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-f16.gguf");

        try (FileChannel channel = FileChannel.open(fixturePath.get(), StandardOpenOption.READ);
                Arena arena = Arena.ofShared()) {
            GGUF fixture = GGUF.read(fixturePath.get());
            float[] pcm = Fixtures.floats(channel, fixture, "pcm");

            TranscriptionModel<?, ?, ?> loaded = Models.loadTranscription(model, arena);
            assertEquals(16_000, loaded.sampleRate());
            Transcription transcription = loaded.transcribe(pcm);
            assertEquals(
                    fixture.getValue(String.class, "fixture.transcript"), transcription.text());

            assertFalse(transcription.tokens().isEmpty());
            Duration previousStart = Duration.ZERO;
            Duration audio = Duration.ofNanos(pcm.length * 1_000_000_000L / loaded.sampleRate());
            for (Transcription.Token token : transcription.tokens()) {
                assertTrue(
                        token.start().compareTo(previousStart) >= 0,
                        "token starts must not go backwards");
                assertTrue(token.end().compareTo(audio) <= 0, "token end past the audio");
                previousStart = token.start();
            }

            // 10 s chunks over the 11 s clip commit mid-utterance, which must not change the text.
            try (var chunks = Fixtures.chunkSeconds(10)) {
                assertEquals(
                        transcription.text(),
                        loaded.transcribe(pcm).text(),
                        "chunked transcription drifted");
            }
        }
    }
}
