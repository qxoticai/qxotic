package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.time.Duration;
import java.util.List;
import org.junit.jupiter.api.Test;

class TranscriptionTest {

    private static Duration ms(long millis) {
        return Duration.ofMillis(millis);
    }

    @Test
    void tokensGroupIntoWordsAtLeadingSpaces() {
        Transcription transcription =
                new Transcription(
                        "And so, my",
                        List.of(
                                new Transcription.Token(" And", ms(200), ms(500), 0.9),
                                new Transcription.Token(" so", ms(500), ms(800), 0.8),
                                new Transcription.Token(",", ms(800), ms(900), 0.6),
                                new Transcription.Token(" my", ms(1000), ms(1200), 1.0)));
        List<Transcription.Word> words = transcription.words();
        assertEquals(3, words.size());
        assertEquals(new Transcription.Word("And", ms(200), ms(500), 0.9), words.get(0));
        assertEquals(new Transcription.Word("so,", ms(500), ms(900), 0.6), words.get(1));
        assertEquals(new Transcription.Word("my", ms(1000), ms(1200), 1.0), words.get(2));
    }

    @Test
    void noTokensMeansNoWords() {
        assertTrue(Transcription.empty().words().isEmpty());
        assertTrue(Transcription.empty().text().isEmpty());
        assertTrue(Transcription.empty().tokens().isEmpty());
    }

    @Test
    void invalidTokensAreRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new Transcription.Token("x", ms(1000), ms(500), 1.0));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Transcription.Token("x", ms(0), ms(500), 1.5));
        assertThrows(
                IllegalArgumentException.class,
                () -> new Transcription.Token("x", ms(-1), ms(500), 1.0));
    }
}
