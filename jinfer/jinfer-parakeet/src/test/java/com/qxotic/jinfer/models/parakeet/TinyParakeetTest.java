package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.chat.Models;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The tiny checkpoint loads through every entry point and runs the whole pipeline. */
class TinyParakeetTest {

    @TempDir static Path dir;
    private static Arena arena;
    private static Parakeet parakeet;

    @BeforeAll
    static void load() throws IOException {
        arena = Arena.ofShared();
        parakeet = TinyParakeet.load(dir, arena);
    }

    @AfterAll
    static void unload() {
        if (arena != null) arena.close();
    }

    @Test
    void describesItself() {
        Parakeet.Configuration configuration = parakeet.configuration();
        assertEquals(TinyParakeet.SAMPLE_RATE, parakeet.sampleRate());
        assertEquals(TinyParakeet.D_MODEL, configuration.dModel());
        assertEquals(TinyParakeet.LAYERS, configuration.layers());
        assertEquals(TinyParakeet.PIECES.length, configuration.vocabSize());
        assertEquals(Duration.ofMillis(80), configuration.frame()); // 160-sample hop x8, 16 kHz
    }

    @Test
    void theProviderDispatchesToIt() throws IOException {
        Path file = TinyParakeet.write(dir.resolve("dispatch.gguf"), 7);
        try (Arena dispatchArena = Arena.ofShared()) {
            assertInstanceOf(Parakeet.class, Models.loadTranscription(file, dispatchArena));
        }
    }

    @Test
    void transcribesWithWellFormedTokens() {
        float[] pcm = TinyParakeet.noise(3, 1);
        Transcription transcription = parakeet.transcribe(pcm);
        Duration audio = Duration.ofNanos(pcm.length * 1_000_000_000L / parakeet.sampleRate());
        Duration previous = Duration.ZERO;
        for (Transcription.Token token : transcription.tokens()) {
            assertTrue(token.start().compareTo(previous) >= 0, "starts are monotone");
            assertTrue(token.end().compareTo(audio) <= 0, "tokens end inside the audio");
            assertTrue(token.confidence() >= 0 && token.confidence() <= 1, "confidence in [0,1]");
            previous = token.start();
        }
    }

    /**
     * Bit-identical once warm. The cold first call may differ in the last float bits (interpreted
     * vs JIT-compiled vector code rounds differently), never in text or timing.
     */
    @Test
    void isDeterministic() {
        float[] pcm = TinyParakeet.noise(3, 2);
        Transcription cold = parakeet.transcribe(pcm);
        Transcription warm = parakeet.transcribe(pcm), again = parakeet.transcribe(pcm);
        assertEquals(warm.text(), again.text());
        assertEquals(warm.tokens(), again.tokens()); // records: every field, bit for bit
        Fixtures.assertSameTranscript(warm, cold, "cold");
    }
}
