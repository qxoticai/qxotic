package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Oracle-free invariants over the real model: properties that must hold for any correct
 * transcription, so they catch retiming, normalization and windowing regressions without a
 * parakeet.cpp trace. The model and PCM load once for the class; each test is one or two decodes.
 */
class ParakeetInvariantsTest {

    private static Arena arena;
    private static Parakeet parakeet;
    private static float[] pcm;
    private static Transcription offline;

    @BeforeAll
    static void load() throws IOException {
        Optional<Path> fixturePath = Fixtures.fixture("tdt-0.6b-v3-q8_0-jfk.fixture.gguf");
        assumeTrue(fixturePath.isPresent(), "parakeet fixture not checked out");
        Path model = TestModels.require("mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf");
        try (FileChannel channel = FileChannel.open(fixturePath.get(), StandardOpenOption.READ)) {
            pcm = Fixtures.floats(channel, GGUF.read(fixturePath.get()), "pcm");
        }
        arena = Arena.ofShared();
        parakeet = Fixtures.load(model, arena);
        offline = parakeet.transcribe(pcm);
    }

    @AfterAll
    static void unload() {
        if (arena != null) arena.close();
    }

    @Test
    void tokenSpansAreSane() {
        assertFalse(offline.tokens().isEmpty(), "the reference clip decodes to tokens");
        Duration audio = Duration.ofNanos(pcm.length * 1_000_000_000L / parakeet.sampleRate());
        Duration previousStart = Duration.ZERO;
        StringBuilder joined = new StringBuilder();
        for (Transcription.Token token : offline.tokens()) {
            assertTrue(token.start().compareTo(previousStart) >= 0, "token starts are monotone");
            assertTrue(token.end().compareTo(audio) <= 0, "token ends inside the audio");
            previousStart = token.start();
            joined.append(token.text());
        }
        // text() is the concatenated token texts minus the SentencePiece leading space
        String concatenated = joined.toString();
        assertEquals(
                concatenated.startsWith(" ") ? concatenated.substring(1) : concatenated,
                offline.text());
    }

    /**
     * Log-mel plus per-feature normalization cancels a constant gain (up to the log-zero guard), so
     * halving the input must not change what is heard.
     */
    @Test
    void constantGainLeavesTheTranscriptAlone() {
        float[] halved = new float[pcm.length];
        for (int i = 0; i < pcm.length; i++) halved[i] = pcm[i] * 0.5f;
        assertEquals(offline.text(), parakeet.transcribe(halved).text());
    }

    /**
     * Prepending silence shifts every timestamp by the silence, and nothing else: 20480 samples is
     * exactly 16 encoder frames, so the alignment is frame-exact up to the normalization drift the
     * extra silent frames introduce (bounded here at two frames).
     */
    @Test
    void leadingSilenceShiftsTimestamps() {
        int pad = 20_480; // 1.28 s
        Duration shift = Duration.ofNanos(pad * 1_000_000_000L / parakeet.sampleRate());
        float[] padded = new float[pad + pcm.length];
        System.arraycopy(pcm, 0, padded, pad, pcm.length);
        Transcription shifted = parakeet.transcribe(padded);
        assertEquals(offline.text(), shifted.text());
        Duration tolerance = parakeet.configuration().frame().multipliedBy(2);
        for (int i = 0; i < offline.tokens().size(); i++) {
            Duration expected = offline.tokens().get(i).start().plus(shift);
            Duration actual = shifted.tokens().get(i).start();
            assertTrue(
                    expected.minus(actual).abs().compareTo(tolerance) <= 0,
                    "token " + i + " starts at " + actual + ", expected " + expected);
        }
    }

    /** The feed pattern never changes the transcript. */
    @Test
    void feedPatternNeverChangesTheTranscript() {
        Transcription whole = Fixtures.streamedWhole(parakeet, pcm);
        for (long seed : new long[] {42, 4242}) {
            Transcription streamed = Fixtures.streamed(parakeet, pcm, new Random(seed), 40_000, 0);
            Fixtures.assertSameTranscript(whole, streamed, "seed " + seed);
        }
    }

    /**
     * The greedy TDT decoder has an absorbing trap: after some sentence-final periods the
     * prediction state predicts blank forever (byte-identical in NeMo and parakeet.cpp). The
     * fixture is 90 s of LibriVox English (public domain) whose single-window decode hits the trap
     * twice; the blank watchdog resets the prediction state and decoding resumes. Without it the
     * decode dies at the first trap (~18 s, 59 tokens), so tokens near the end are the watchdog's.
     * The trap needs the accumulated prediction state: a 30 s excerpt around it decodes clean,
     * which is why this fixture is the whole clip.
     */
    @Test
    void watchdogEscapesTheDecoderCollapse() throws IOException {
        Optional<Path> clip = Fixtures.fixture("collapse-trap-en.wav");
        assumeTrue(clip.isPresent(), "collapse-trap clip not checked out");
        float[] trap = wav16(clip.get());
        // one window: the watchdog alone must escape the trap
        try (var chunks = Fixtures.chunkSeconds(120)) {
            Transcription single = parakeet.transcribe(trap);
            assertTrue(single.tokens().size() >= 250, "collapsed: " + single.tokens().size());
            Duration last = single.tokens().getLast().start();
            assertTrue(
                    last.compareTo(Duration.ofSeconds(80)) > 0,
                    "nothing decoded after the trap: last token at " + last);
        }
    }

    /** Canonical 16-bit mono little-endian WAV, which is what the fixture is. */
    private static float[] wav16(Path wav) throws IOException {
        byte[] bytes = Files.readAllBytes(wav);
        ByteBuffer buffer =
                ByteBuffer.wrap(bytes, 44, bytes.length - 44).order(ByteOrder.LITTLE_ENDIAN);
        float[] pcm = new float[buffer.remaining() / 2];
        for (int i = 0; i < pcm.length; i++) pcm[i] = buffer.getShort() / 32768f;
        return pcm;
    }
}
