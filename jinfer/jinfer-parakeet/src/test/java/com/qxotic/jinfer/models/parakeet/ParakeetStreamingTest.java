package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionStream;
import com.qxotic.jinfer.media.Media;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The streaming contract on the tiny checkpoint, which emits a word about three times a second on
 * noise: final pieces arrive as chunks commit and join to the same transcript whatever the feeding,
 * the partial is the tail after them, texts join without losing spaces, times stay inside the fed
 * audio, and a finished or closed stream refuses further use. Runs on every checkout.
 */
class ParakeetStreamingTest {

    @TempDir static Path dir;
    private static Arena arena;
    private static Parakeet parakeet;

    /** 35 s of noise: many 2 s chunks commit, and a tail is left for {@code finish}. */
    private static final float[] AUDIO = TinyParakeet.noise(35, 11);

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
    void theStreamExpectsTheModelsSampleRate() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            assertEquals(parakeet.sampleRate(), stream.sampleRate());
        }
    }

    @Test
    void audioShorterThanAChunkAndItsRightContextIsFinalOnlyAtFinish() {
        float[] pcm = TinyParakeet.noise(3.5, 3);
        Transcription oneShot = parakeet.transcribe(pcm);
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            assertEquals(List.of(), stream.feed(pcm, 0, 20_000).tokens());
            assertEquals(List.of(), stream.feed(pcm, 20_000, pcm.length - 20_000).tokens());
            Fixtures.assertSameTranscript(oneShot, stream.finish(), "finish");
        }
    }

    @Test
    void finalPiecesArriveEveryChunk() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            int second = parakeet.sampleRate(), nonEmpty = 0;
            for (int at = 0; at < AUDIO.length; at += second) {
                Transcription piece = stream.feed(AUDIO, at, Math.min(second, AUDIO.length - at));
                if (!piece.tokens().isEmpty()) nonEmpty++;
                // the first chunk commits once 2 s of right context follow it, at 4 s
                if (at + second < 4 * second)
                    assertEquals(List.of(), piece.tokens(), "final before a chunk");
            }
            assertTrue(nonEmpty >= 10, "only " + nonEmpty + " pieces before finish");
            assertFalse(stream.finish().tokens().isEmpty(), "no tail left to finish");
        }
    }

    /** Any chunking, with or without partials polled between feeds, joins to the same tokens. */
    @Test
    void piecesJoinToTheSameTranscriptWhateverTheFeeding() {
        Transcription oneShot = Fixtures.streamedWhole(parakeet, AUDIO);
        assertTrue(oneShot.tokens().size() > 50, "the tiny model went quiet");
        for (long seed = 1; seed <= 4; seed++) {
            int partialOdds = seed % 2 == 0 ? 3 : 0;
            Transcription joined =
                    Fixtures.streamed(parakeet, AUDIO, new Random(seed), 30_000, partialOdds);
            Fixtures.assertSameTranscript(oneShot, joined, "seed " + seed);
        }
    }

    @Test
    void pieceTextsJoinWithoutLosingSpaces() {
        List<Transcription> pieces = new ArrayList<>();
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            int step = 3 * parakeet.sampleRate();
            for (int at = 0; at < AUDIO.length; at += step)
                pieces.add(stream.feed(AUDIO, at, Math.min(step, AUDIO.length - at)));
            pieces.add(stream.finish());
        }
        boolean first = true;
        StringBuilder joined = new StringBuilder();
        for (Transcription piece : pieces) {
            if (piece.text().isEmpty()) continue;
            assertEquals(!first, piece.text().startsWith(" "), "'" + piece.text() + "'");
            joined.append(piece.text());
            first = false;
        }
        assertEquals(Fixtures.streamedWhole(parakeet, AUDIO).text(), joined.toString());
    }

    /**
     * After every feed, the partial starts after the last final token, stays inside the fed audio,
     * and keeps its leading space only once final text exists.
     */
    @Test
    void thePartialIsTheTailAfterTheLastFinalPiece() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            Fixtures.Pieces pieces = new Fixtures.Pieces();
            int step = 2 * parakeet.sampleRate(), fed = 0;
            for (int at = 0; at < AUDIO.length; at += step) {
                int length = Math.min(step, AUDIO.length - at);
                pieces.add(stream.feed(AUDIO, at, length));
                fed += length;
                Transcription tail = stream.partial();
                List<Transcription.Token> finals = pieces.joined().tokens();
                assertInsideAudio(tail, fed);
                if (tail.tokens().isEmpty()) continue;
                if (!finals.isEmpty())
                    assertTrue(
                            tail.tokens().getFirst().start().compareTo(finals.getLast().start())
                                    > 0,
                            "the partial overlaps the final text");
                assertEquals(!finals.isEmpty(), tail.text().startsWith(" "));
            }
        }
    }

    /** Final tokens are in time order across pieces and never end past the audio fed so far. */
    @Test
    void finalTokensAreOrderedAndInsideTheFedAudio() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            List<Transcription> pieces = new ArrayList<>();
            int step = 4 * parakeet.sampleRate();
            for (int at = 0; at < AUDIO.length; at += step) {
                int length = Math.min(step, AUDIO.length - at);
                Transcription piece = stream.feed(AUDIO, at, length);
                assertInsideAudio(piece, at + length);
                pieces.add(piece);
            }
            Transcription last = stream.finish();
            assertInsideAudio(last, AUDIO.length);
            pieces.add(last);
            Duration previous = Duration.ZERO;
            for (Transcription piece : pieces)
                for (Transcription.Token token : piece.tokens()) {
                    assertTrue(token.start().compareTo(previous) >= 0, "order");
                    previous = token.start();
                }
        }
    }

    @Test
    void anEmptyStreamFinishesEmpty() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            assertSame(Transcription.empty(), stream.partial());
            assertSame(Transcription.empty(), stream.finish());
        }
    }

    @Test
    void feedingNothingReturnsNothing() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            assertSame(Transcription.empty(), stream.feed(new float[0]));
            assertSame(Transcription.empty(), stream.feed(AUDIO, 7, 0));
        }
    }

    @Test
    void feedChecksItsArguments() {
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            assertThrows(NullPointerException.class, () -> stream.feed(null, 0, 0));
            assertThrows(IndexOutOfBoundsException.class, () -> stream.feed(AUDIO, -1, 10));
            assertThrows(IndexOutOfBoundsException.class, () -> stream.feed(AUDIO, 0, -1));
            assertThrows(
                    IndexOutOfBoundsException.class, () -> stream.feed(AUDIO, AUDIO.length, 1));
        }
    }

    @Test
    void decodedAudioIsAcceptedOnlyInTheStreamsFormat() {
        float[] pcm = new float[parakeet.sampleRate()];
        for (int i = 0; i < pcm.length; i++) pcm[i] = Math.clamp(AUDIO[i], -1f, 1f);
        try (var state = parakeet.newState();
                var stream = parakeet.stream(state)) {
            stream.feed(new Media.Audio(pcm, parakeet.sampleRate(), 1));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> stream.feed(new Media.Audio(pcm, 8_000, 1)));
            assertThrows(
                    IllegalArgumentException.class,
                    () -> stream.feed(new Media.Audio(pcm, parakeet.sampleRate(), 2)));
        }
    }

    @Test
    void aFinishedStreamRefusesEveryCallButClose() {
        try (var state = parakeet.newState()) {
            TranscriptionStream stream = parakeet.stream(state);
            stream.feed(AUDIO, 0, parakeet.sampleRate());
            stream.finish();
            assertRefusesUse(stream);
            stream.close();
            stream.close(); // idempotent
        }
    }

    @Test
    void aClosedStreamRefusesEveryCallButClose() {
        try (var state = parakeet.newState()) {
            TranscriptionStream stream = parakeet.stream(state);
            stream.feed(AUDIO, 0, parakeet.sampleRate());
            stream.close();
            assertRefusesUse(stream);
            stream.close();
        }
    }

    /** Streams on one state, one after another and even abandoned midway, stay independent. */
    @Test
    void oneStateServesStreamsBackToBack() {
        Transcription oneShot = Fixtures.streamedWhole(parakeet, AUDIO);
        try (var state = parakeet.newState()) {
            try (var abandoned = parakeet.stream(state)) {
                abandoned.feed(AUDIO, 0, 12 * parakeet.sampleRate());
                abandoned.partial();
            }
            for (int round = 0; round < 2; round++) {
                try (var stream = parakeet.stream(state)) {
                    Fixtures.Pieces pieces = new Fixtures.Pieces();
                    pieces.add(stream.feed(AUDIO));
                    pieces.add(stream.finish());
                    Fixtures.assertSameTranscript(oneShot, pieces.joined(), "round " + round);
                }
            }
            Fixtures.assertSameTranscript(
                    parakeet.transcribe(AUDIO),
                    parakeet.transcribe(state, AUDIO),
                    "transcribe after");
        }
    }

    private static void assertRefusesUse(TranscriptionStream stream) {
        assertThrows(IllegalStateException.class, () -> stream.feed(AUDIO, 0, 10));
        assertThrows(IllegalStateException.class, stream::partial);
        assertThrows(IllegalStateException.class, stream::finish);
    }

    private static void assertInsideAudio(Transcription transcription, int fedSamples) {
        Duration fed = Duration.ofNanos(fedSamples * 1_000_000_000L / parakeet.sampleRate());
        for (Transcription.Token token : transcription.tokens())
            assertTrue(token.end().compareTo(fed) <= 0, token + " ends past " + fed);
    }
}
