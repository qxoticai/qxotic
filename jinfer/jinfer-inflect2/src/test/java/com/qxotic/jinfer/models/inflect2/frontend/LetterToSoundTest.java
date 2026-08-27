// Golden pairs and invariants for the letter-to-sound fallback.
package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.models.inflect2.Symbols;
import org.junit.jupiter.api.Test;

class LetterToSoundTest {

    @Test
    void goldenWords() {
        assertEquals("ˈkæt", LetterToSound.guess("cat"));
        assertEquals("ˈsplæʃ", LetterToSound.guess("splash"));
        assertEquals("ˈθɪŋs", LetterToSound.guess("things"));
        assertEquals("ˈʧʌŋk", LetterToSound.guess("chunk"));
        assertEquals("ˈmeɪk", LetterToSound.guess("make")); // magic e
        assertEquals("ˈhæpi", LetterToSound.guess("happy")); // double consonant, final y
        assertEquals("ˈʤʌʤ", LetterToSound.guess("judge"));
        assertEquals("ˈjɛs", LetterToSound.guess("yes")); // consonantal y
        assertEquals("ˈsɪti", LetterToSound.guess("city")); // soft c
        assertEquals("ˈʤɪnfɜɹ", LetterToSound.guess("Jinfer"));
    }

    @Test
    void mixedCaseAndNoiseAreHandled() {
        assertEquals(LetterToSound.guess("cat"), LetterToSound.guess("CAT"));
        assertEquals(LetterToSound.guess("cat"), LetterToSound.guess("Cat!"));
        assertEquals("", LetterToSound.guess("..."));
        assertEquals("", LetterToSound.guess(""));
    }

    @Test
    void everyGuessIsSpeakableByTheModel() {
        // any word gets a pronunciation, and every emitted symbol is in the model's table
        for (String word :
                new String[] {
                    "blorp",
                    "unboxing",
                    "kvetch",
                    "photosynthesis",
                    "yyz",
                    "schmidt",
                    "qwirk",
                    "throughout",
                    "knight",
                    "psychology"
                }) {
            String guess = LetterToSound.guess(word);
            assertTrue(guess.codePoints().anyMatch(cp -> Symbols.idOf(cp) != 0), word);
            for (int cp : guess.codePoints().toArray())
                assertTrue(Symbols.idOf(cp) != 0, word + " emits a symbol outside the table");
        }
    }
}
