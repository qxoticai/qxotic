package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import org.junit.jupiter.api.Test;

class TranscriptionTest {

    @Test
    void tokensGroupIntoWordsAtLeadingSpaces() {
        Transcription transcription =
                new Transcription(
                        "And so, my",
                        List.of(
                                new Transcription.Token(" And", 0.2, 0.5, 0.9),
                                new Transcription.Token(" so", 0.5, 0.8, 0.8),
                                new Transcription.Token(",", 0.8, 0.9, 0.6),
                                new Transcription.Token(" my", 1.0, 1.2, 1.0)));
        List<Transcription.Word> words = transcription.words();
        assertEquals(3, words.size());
        assertEquals(new Transcription.Word("And", 0.2, 0.5, 0.9), words.get(0));
        assertEquals(new Transcription.Word("so,", 0.5, 0.9, 0.6), words.get(1));
        assertEquals(new Transcription.Word("my", 1.0, 1.2, 1.0), words.get(2));
    }

    @Test
    void noTokensMeansNoWords() {
        assertTrue(new Transcription("", List.of()).words().isEmpty());
    }

    @Test
    void invalidTokensAreRefused() {
        assertThrows(
                IllegalArgumentException.class, () -> new Transcription.Token("x", 1.0, 0.5, 1.0));
        assertThrows(
                IllegalArgumentException.class, () -> new Transcription.Token("x", 0.0, 0.5, 1.5));
    }
}
