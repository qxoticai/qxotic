package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Pins the punctuation-run split, not the phonemes: espeak's IPA is version-dependent, so the
 * durable contract is that every mark survives exactly once in source order and that every
 * punctuation-free span reaches espeak - including the trailing one, which the loop is one missing
 * flush away from dropping.
 *
 * <p>Skips when espeak-ng is absent; the lexicon front end is tested separately.
 */
class EspeakPhonemizerTest {

    private static Phonemizer espeak() {
        Phonemizer espeak = EspeakPhonemizer.tryCreate();
        Assumptions.assumeTrue(espeak != null, "espeak-ng is not installed");
        return espeak;
    }

    /** The symbols of {@code tokens} that are punctuation, in order - blanks and phonemes out. */
    private static List<Integer> marks(int[] tokens) {
        List<Integer> marks = new ArrayList<>();
        for (int token : tokens) {
            for (char mark : new char[] {',', '.', ';', '!', '?', ':'}) {
                if (token == Symbols.idOf(mark)) marks.add(token);
            }
        }
        return marks;
    }

    @Test
    void everyMarkSurvivesOnceInSourceOrder() throws IOException {
        int[] tokens = espeak().phonemize("Yes, of course; that is right. Is it?");
        assertEquals(
                List.of(
                        Symbols.idOf(','),
                        Symbols.idOf(';'),
                        Symbols.idOf('.'),
                        Symbols.idOf('?')),
                marks(tokens));
    }

    @Test
    void theTrailingRunIsFlushed() throws IOException {
        // no punctuation at all: the whole text is one open run, emitted only by the final flush
        int[] tokens = espeak().phonemize("one two three");
        assertTrue(tokens.length > 6, "a three-word run must phonemize to more than blanks");
        assertTrue(marks(tokens).isEmpty(), "nothing punctuates this");
    }

    @Test
    void aRunIsPhonemizedAsOneSpan() throws IOException {
        Phonemizer espeak = espeak();
        // espeak reads context: per word it stamps a primary stress on every function word, so a
        // batched run is NOT the concatenation of its words. If this ever matches, the split
        // regressed to one subprocess per word.
        int[] run = espeak.phonemize("the cat can see the dog");
        int[] words = concat(espeak, "the", "cat", "can", "see", "the", "dog");
        assertTrue(run.length != words.length || !java.util.Arrays.equals(run, words));
    }

    private static int[] concat(Phonemizer espeak, String... words) throws IOException {
        int[] tokens = new int[0];
        for (String word : words) {
            int[] one = espeak.phonemize(word);
            int[] grown = java.util.Arrays.copyOf(tokens, tokens.length + one.length);
            System.arraycopy(one, 0, grown, tokens.length, one.length);
            tokens = grown;
        }
        return tokens;
    }
}
