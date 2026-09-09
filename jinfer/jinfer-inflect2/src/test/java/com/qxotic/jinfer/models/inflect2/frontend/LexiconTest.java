// The layered front end: lexicon first, espeak per uncovered run, letter-to-sound as the last rung.
// Runs only: punctuation and the symbol table are the core RunPhonemizer's, tested there.
package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class LexiconTest {

    private static final Map<String, String> LEXICON =
            Map.of("hello", "həloʊ", "world", "wɜɹld", "z", "z", "d", "d");

    @Test
    void knownWordsUseTheLexicon() {
        var calls = new AtomicInteger();
        Lexicon lexicon =
                Lexicon.of(
                        LEXICON,
                        run -> {
                            calls.incrementAndGet();
                            return "stʌf";
                        });
        assertEquals("həloʊ wɜɹld", lexicon.ipa("hello world"));
        assertEquals(0, calls.get(), "a fully covered run must not touch the fallback");
        assertEquals("həloʊ", lexicon.ipa("Hello"), "only the lexicon key is lowercased");
    }

    @Test
    void anInflectionIsTheStemPlusItsSpokenSuffix() {
        Lexicon lexicon = Lexicon.of(LEXICON, null);
        assertEquals("wɜɹldz", lexicon.ipa("worlds"));
        assertEquals("həloʊd", lexicon.ipa("helloed"));
    }

    @Test
    void aRunWithAnUnknownWordGoesToTheFallbackWhole() {
        var seen = new AtomicInteger();
        Lexicon lexicon =
                Lexicon.of(
                        LEXICON,
                        run -> {
                            seen.incrementAndGet();
                            assertEquals("hello blorp", run); // the whole run, not the word
                            return "həloʊ blɔɹp";
                        });
        assertEquals("həloʊ blɔɹp", lexicon.ipa("hello blorp"));
        assertEquals(1, seen.get());
    }

    @Test
    void theFallbackSeesTheSpellingTheWriterUsed() {
        // Only the lexicon KEY is lowercased. espeak reads capitals as information - it says
        // "GraalVM" as "graal vee em" and the lowercased "graalvm" as one mangled word - so
        // handing it a flattened run threw away the only clue it had, and the VM went silent.
        var seen = new AtomicInteger();
        Lexicon lexicon =
                Lexicon.of(
                        LEXICON,
                        run -> {
                            seen.incrementAndGet();
                            assertEquals("Hello GraalVM", run);
                            return "həloʊ";
                        });
        lexicon.ipa("Hello GraalVM");
        assertEquals(1, seen.get());
    }

    @Test
    void withoutAFallbackUnknownWordsAreGuessedByRule() {
        Lexicon lexicon = Lexicon.of(LEXICON, null);
        assertEquals("həloʊ " + LetterToSound.guess("blorp"), lexicon.ipa("hello blorp"));
    }

    @Test
    void lookupIsLocaleIndependent() {
        // under a Turkish default locale toLowerCase() turns "I" into dotless i, misses the
        // lexicon, and the letter-to-sound guess of a non-ASCII letter was empty: a silent word
        Locale saved = Locale.getDefault();
        Locale.setDefault(Locale.forLanguageTag("tr-TR"));
        try {
            Lexicon lexicon = Lexicon.of(LEXICON, null);
            assertEquals(lexicon.ipa("hello"), lexicon.ipa("HELLO"));
            assertEquals(lexicon.ipa("hi"), lexicon.ipa("HI"));
        } finally {
            Locale.setDefault(saved);
        }
    }

    @Test
    void aTruncatedLexiconIsReportedAsAnIoError() throws IOException {
        var path = Files.createTempFile("truncated-lexicon", ".bin");
        Files.write(path, new byte[] {'I', 'V', 'L', '2'});

        IOException error = assertThrows(IOException.class, () -> Lexicon.read(path, null));

        assertTrue(error.getMessage().contains("malformed"), error.getMessage());
    }
}
