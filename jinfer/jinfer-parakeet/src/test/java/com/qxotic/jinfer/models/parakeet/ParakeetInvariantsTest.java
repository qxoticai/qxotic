package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.kernels.ModelLoader;
import com.qxotic.jinfer.testkit.TestModels;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Optional;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * Oracle-free invariants over the real model: properties that must hold for ANY correct
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
        try (FileChannel channel = FileChannel.open(model, StandardOpenOption.READ)) {
            parakeet =
                    Parakeet.load(channel, ModelLoader.readGguf(channel, model.toString()), arena);
        }
        offline = parakeet.transcribe(pcm);
    }

    @AfterAll
    static void unload() {
        if (arena != null) arena.close();
    }

    @Test
    void tokenSpansAreSane() {
        assertFalse(offline.tokens().isEmpty(), "the reference clip decodes to tokens");
        double duration = (double) pcm.length / parakeet.sampleRate();
        double previousStart = 0;
        StringBuilder joined = new StringBuilder();
        for (Transcription.Token token : offline.tokens()) {
            assertTrue(token.start() >= previousStart, "token starts must be monotone");
            assertTrue(token.end() <= duration + 1e-9, "token ends inside the audio");
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
        double shift = (double) pad / parakeet.sampleRate();
        float[] padded = new float[pad + pcm.length];
        System.arraycopy(pcm, 0, padded, pad, pcm.length);
        Transcription shifted = parakeet.transcribe(padded);
        assertEquals(offline.text(), shifted.text());
        double tolerance = 2 * parakeet.configuration().frameSeconds();
        for (int i = 0; i < offline.tokens().size(); i++) {
            double expected = offline.tokens().get(i).start() + shift;
            double actual = shifted.tokens().get(i).start();
            assertEquals(expected, actual, tolerance, "token " + i + " start");
        }
    }

    /** The feed pattern must never change the transcript - the streaming determinism law. */
    @Test
    void feedPatternNeverChangesTheTranscript() {
        for (long seed : new long[] {42, 4242}) {
            assertEquals(offline.text(), randomFeed(pcm, seed).text(), "seed " + seed);
        }

        // and across window commits: doubled audio over 10 s windows forces mid-stream commits
        float[] doubled = new float[pcm.length * 2];
        System.arraycopy(pcm, 0, doubled, 0, pcm.length);
        System.arraycopy(pcm, 0, doubled, pcm.length, pcm.length);
        System.setProperty("jinfer.parakeet.chunkSeconds", "10");
        try {
            String windowed = parakeet.transcribe(doubled).text();
            assertEquals(windowed, randomFeed(doubled, 7).text(), "windowed, seed 7");
        } finally {
            System.clearProperty("jinfer.parakeet.chunkSeconds");
        }
    }

    /**
     * The greedy TDT decoder has an absorbing trap: after some sentence-final periods the
     * prediction state predicts blank forever (byte-identical in NeMo and parakeet.cpp). The
     * fixture is 90 s of LibriVox English (public domain) whose SINGLE-window decode hits the trap
     * twice; the blank watchdog rewinds with a fresh state and decoding resumes. Without it the
     * decode dies at the first trap (~18 s, 59 tokens) - so tokens near the end ARE the watchdog.
     * The trap needs the accumulated prediction state: a 30 s excerpt around it decodes clean,
     * which is why this fixture is the whole clip.
     */
    @Test
    void watchdogEscapesTheDecoderCollapse() throws IOException {
        Optional<Path> clip = Fixtures.fixture("collapse-trap-en.wav");
        assumeTrue(clip.isPresent(), "collapse-trap clip not checked out");
        float[] trap = wav16(clip.get());
        System.setProperty("jinfer.parakeet.chunkSeconds", "120"); // one window: no seam rescue
        try {
            Transcription single = parakeet.transcribe(trap);
            assertTrue(single.tokens().size() >= 250, "collapsed: " + single.tokens().size());
            double last = single.tokens().getLast().start();
            assertTrue(last > 80, "nothing decoded after the trap: last token at " + last + " s");
        } finally {
            System.clearProperty("jinfer.parakeet.chunkSeconds");
        }
    }

    /** Canonical 16-bit mono little-endian WAV, which is what the fixture is. */
    private static float[] wav16(Path wav) throws IOException {
        byte[] bytes = java.nio.file.Files.readAllBytes(wav);
        java.nio.ByteBuffer buffer =
                java.nio.ByteBuffer.wrap(bytes, 44, bytes.length - 44)
                        .order(java.nio.ByteOrder.LITTLE_ENDIAN);
        float[] pcm = new float[buffer.remaining() / 2];
        for (int i = 0; i < pcm.length; i++) pcm[i] = buffer.getShort() / 32768f;
        return pcm;
    }

    private static Transcription randomFeed(float[] audio, long seed) {
        Random random = new Random(seed);
        try (Parakeet.State state = parakeet.newState()) {
            TranscriptionStream stream = parakeet.stream(state);
            int at = 0;
            while (at < audio.length) {
                int chunk = Math.min(1 + random.nextInt(40_000), audio.length - at);
                stream.feed(audio, at, chunk);
                at += chunk;
            }
            return stream.finish();
        }
    }
}
