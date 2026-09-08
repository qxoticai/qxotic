package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.SpeechOptions;
import java.util.List;
import org.junit.jupiter.api.Test;

final class KokoroTTSTest {

    @Test
    void splitsLongTextAtSentenceAndWordBoundaries() {
        assertEquals(List.of("One.", "Two?"), KokoroTTS.chunks(" One.  Two? "));
        assertEquals(List.of("One.", "Two?"), KokoroTTS.chunks("One.\u2003Two?"));
        String words = "word ".repeat(50).trim();
        List<String> chunks = KokoroTTS.chunks(words);
        assertEquals(words, String.join(" ", chunks));
        for (String chunk : chunks) assertEquals(true, chunk.length() <= 200);
    }

    @Test
    void doesNotSplitSurrogatePairs() {
        String text = "a".repeat(199) + "\uD83D\uDE00" + "b";

        assertEquals(List.of("a".repeat(199), "\uD83D\uDE00b"), KokoroTTS.chunks(text));
    }

    @Test
    void rejectsNonFiniteSpeed() {
        assertThrows(
                IllegalArgumentException.class,
                () -> KokoroTTS.speed(SpeechOptions.speed(Double.NaN)));
    }
}
