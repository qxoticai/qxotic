// The layered frontend: lexicon first, espeak per uncovered run, letter-to-sound as the last rung.
package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.IOException;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class LexiconFallbackTest {

    private static final int SPACE = Symbols.idOf(' ');
    private static final Map<String, int[]> LEXICON =
            Map.of(
                    "hello", Symbols.toRaw("həloʊ"),
                    "world", Symbols.toRaw("wɜɹld"));

    private static int[] concat(int[]... blocks) {
        int total = 0;
        for (int[] b : blocks) total += b.length;
        int[] out = new int[total];
        int at = 0;
        for (int[] b : blocks) {
            System.arraycopy(b, 0, out, at, b.length);
            at += b.length;
        }
        return out;
    }

    @Test
    void knownWordsUseTheLexicon() throws IOException {
        var calls = new AtomicInteger();
        var p = LexiconPhonemizer.of(LEXICON, run -> "ignored" /* and counted */);
        Phonemizer counting =
                LexiconPhonemizer.of(
                        LEXICON,
                        run -> {
                            calls.incrementAndGet();
                            return "stʌf";
                        });
        assertArrayEquals(
                Symbols.blankIntersperse(
                        concat(Symbols.toRaw("həloʊ"), new int[] {SPACE}, Symbols.toRaw("wɜɹld"))),
                counting.phonemize("hello world"));
        assertEquals(0, calls.get(), "a fully covered run must not touch the fallback");
        assertArrayEquals(Symbols.blankIntersperse(Symbols.toRaw("həloʊ")), p.phonemize("Hello"));
    }

    @Test
    void aRunWithAnUnknownWordGoesToTheFallbackWhole() throws IOException {
        var seen = new AtomicInteger();
        Phonemizer p =
                LexiconPhonemizer.of(
                        LEXICON,
                        run -> {
                            seen.incrementAndGet();
                            assertEquals("hello blorp", run); // the whole run, not the word
                            return "həloʊ blɔɹp";
                        });
        assertArrayEquals(
                Symbols.blankIntersperse(Symbols.toRaw("həloʊ blɔɹp")), p.phonemize("hello blorp"));
        assertEquals(1, seen.get());
    }

    @Test
    void theFallbackSeesTheSpellingTheWriterUsed() throws IOException {
        // Only the lexicon KEY is lowercased. espeak reads capitals as information - it says
        // "GraalVM" as "graal vee em" and the lowercased "graalvm" as one mangled word - so
        // handing it a flattened run threw away the only clue it had, and the VM went silent.
        var seen = new AtomicInteger();
        Phonemizer p =
                LexiconPhonemizer.of(
                        LEXICON,
                        run -> {
                            seen.incrementAndGet();
                            assertEquals("Hello GraalVM", run);
                            return "həloʊ";
                        });
        p.phonemize("Hello GraalVM");
        assertEquals(1, seen.get());
    }

    @Test
    void withoutAFallbackUnknownWordsAreGuessedByRule() throws IOException {
        Phonemizer p = LexiconPhonemizer.of(LEXICON, null);
        int[] expected =
                Symbols.blankIntersperse(
                        concat(
                                Symbols.toRaw("həloʊ"),
                                new int[] {SPACE},
                                Symbols.toRaw(LetterToSound.guess("blorp"))));
        assertArrayEquals(expected, p.phonemize("hello blorp"));
    }

    @Test
    void punctuationKeepsItsSymbolsAndClosesRuns() throws IOException {
        Phonemizer p = LexiconPhonemizer.of(LEXICON, null);
        int[] expected =
                Symbols.blankIntersperse(
                        concat(
                                Symbols.toRaw("həloʊ"),
                                new int[] {SPACE, Symbols.idOf(','), SPACE},
                                Symbols.toRaw("wɜɹld"),
                                new int[] {SPACE, Symbols.idOf('.')}));
        assertArrayEquals(expected, p.phonemize("hello, world."));
    }

    @Test
    void emptyAndPunctuationOnlyInputStillWork() throws IOException {
        Phonemizer p = LexiconPhonemizer.of(LEXICON, null);
        assertArrayEquals(Symbols.blankIntersperse(new int[0]), p.phonemize(""));
        assertArrayEquals(
                Symbols.blankIntersperse(new int[] {Symbols.idOf('.')}), p.phonemize("."));
    }

    @Test
    void lookupIsLocaleIndependent() throws IOException {
        // under a Turkish default locale toLowerCase() turns "I" into dotless i, misses the
        // lexicon, and the letter-to-sound guess of a non-ASCII letter was empty: a silent word
        Locale saved = Locale.getDefault();
        Locale.setDefault(Locale.forLanguageTag("tr-TR"));
        try {
            Phonemizer p = LexiconPhonemizer.of(LEXICON, null);
            assertArrayEquals(p.phonemize("hello"), p.phonemize("HELLO"));
            assertArrayEquals(p.phonemize("hi"), p.phonemize("HI"));
        } finally {
            Locale.setDefault(saved);
        }
    }

    @Test
    void anOpeningQuoteIsAMarkNotPartOfTheWord() throws IOException {
        // "\"hello" missed the lexicon (and reached espeak whole); the opening quote is a symbol
        // before the word, spaced like the closing one already was after it
        Phonemizer p = LexiconPhonemizer.of(LEXICON, null);
        int[] quote = Symbols.toRaw("\"");
        int[] space = Symbols.toRaw(" ");
        assertArrayEquals(
                Symbols.blankIntersperse(
                        concat(quote, space, Symbols.toRaw("həloʊ"), space, quote)),
                p.phonemize("\"hello\""));
    }
}
