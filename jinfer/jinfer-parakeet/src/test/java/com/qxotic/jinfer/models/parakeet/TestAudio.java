package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;
import java.util.Optional;

/**
 * The end-to-end clips in {@code test-fixtures/parakeet/}, fetched by {@code make test-fixtures},
 * decoded, resampled where needed, to the model's 16 kHz mono: sherpa-onnx's Parakeet test speech
 * ({@code en}, {@code de}, {@code es}, {@code fr}, and {@code en0}, byte-identical to
 * parakeet.cpp's {@code speech.wav}) and three LibriSpeech test-clean utterances ({@code ls0},
 * {@code ls1}, and {@code ls8k} at 8 kHz).
 */
final class TestAudio {

    private TestAudio() {}

    static float[] clip(String name) {
        Optional<Path> wav = Fixtures.fixture(name + ".wav");
        assumeTrue(wav.isPresent(), name + ".wav not fetched: run make test-fixtures");
        try {
            Media.Audio audio = AudioCodec.load(wav.get());
            assertEquals(16_000, audio.sampleRate(), "decoders resample to 16 kHz");
            assertEquals(1, audio.channels(), "decoders downmix to mono");
            return audio.pcm();
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
    }
}
