package com.qxotic.jinfer;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.util.ArrayList;
import java.util.List;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.Test;

/**
 * Pins the run split and the table, not any phonemes: the grapheme-to-phoneme step is a fake that
 * brackets what it was given, so what reaches it and what comes back are both visible.
 */
final class RunPhonemizerTest {

    /** Slot 0 is the pad, as in every table of this family; the rest is what the tests need. */
    private static final List<String> SYMBOLS =
            table(
                    "",
                    " ",
                    ",",
                    ".",
                    ";",
                    "!",
                    "?",
                    "'",
                    "\"",
                    "(",
                    ")",
                    "[",
                    "]",
                    "-",
                    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ");

    private static List<String> table(String... entries) {
        List<String> out = new ArrayList<>();
        for (String entry : entries) {
            if (entry.length() <= 1) out.add(entry);
            else entry.codePoints().forEach(cp -> out.add(Character.toString(cp)));
        }
        return List.copyOf(out);
    }

    private static int id(String symbol) {
        return SYMBOLS.indexOf(symbol);
    }

    private static int[] ids(String text) {
        return text.chars().map(c -> id(Character.toString(c))).toArray();
    }

    /** The runs the G2P saw, in order. */
    private static final class Recorder implements UnaryOperator<String> {
        final List<String> runs = new ArrayList<>();

        @Override
        public String apply(String run) {
            runs.add(run);
            return "[" + run + "]";
        }
    }

    @Test
    void marksSplitRunsAndSurviveOnceInSourceOrder() {
        Recorder g2p = new Recorder();
        int[] out = Phonemizer.ipa(SYMBOLS, g2p).phonemize("Yes, of course; that is right. Is it?");

        assertEquals(List.of("Yes", "of course", "that is right", "Is it"), g2p.runs);
        assertArrayEquals(ids("[Yes] , [of course] ; [that is right] . [Is it] ?"), out);
    }

    @Test
    void theTrailingRunIsFlushed() {
        Recorder g2p = new Recorder();
        Phonemizer.ipa(SYMBOLS, g2p).phonemize("one two three");
        assertEquals(List.of("one two three"), g2p.runs);
    }

    @Test
    void wordInternalPunctuationStaysInTheWord() {
        Recorder g2p = new Recorder();
        Phonemizer.ipa(SYMBOLS, g2p).phonemize("don't say 3.5 or twenty-one, dogs' bones");
        assertEquals(List.of("don't say 3.5 or twenty-one", "dogs' bones"), g2p.runs);
    }

    @Test
    void anOpeningQuoteComesBeforeItsWord() {
        Recorder g2p = new Recorder();
        int[] out = Phonemizer.ipa(SYMBOLS, g2p).phonemize("\"hello\" (there)");
        assertEquals(List.of("hello", "there"), g2p.runs);
        assertArrayEquals(ids("\" [hello] \" ( [there] )"), out);
    }

    @Test
    void allPunctuationTokensAreMarksNotRuns() {
        Recorder g2p = new Recorder();
        int[] out = Phonemizer.ipa(SYMBOLS, g2p).phonemize("well ... no");
        assertEquals(List.of("well", "no"), g2p.runs);
        assertArrayEquals(ids("[well] ... [no]"), out);
    }

    @Test
    void theWritersCaseAndSpacingReachTheG2pAsOneRun() {
        Recorder g2p = new Recorder();
        Phonemizer.ipa(SYMBOLS, g2p).phonemize("  GraalVM   is\tfast  ");
        assertEquals(List.of("GraalVM is fast"), g2p.runs);
    }

    @Test
    void ipaTheTableLacksIsDroppedNotPadded() {
        int[] out = Phonemizer.ipa(SYMBOLS, run -> "aéb").phonemize("x");
        assertArrayEquals(ids("ab"), out);
    }

    @Test
    void aDroppedMarkLeavesNoTrace() {
        // '%' is not on the table: "36 % on" must not become "36  on" with two separators
        Recorder g2p = new Recorder();
        int[] out = Phonemizer.ipa(SYMBOLS, g2p).phonemize("36% on");
        assertEquals(List.of("36", "on"), g2p.runs);
        assertArrayEquals(ids("[36] [on]"), out);
    }

    @Test
    void nothingPronounceableIsEmptyNotNull() {
        assertArrayEquals(new int[0], Phonemizer.ipa(SYMBOLS, run -> "").phonemize("   "));
    }

    @Test
    void aSparseTableMapsHighCodePointsAndSkipsEmptySlots() {
        List<String> sparse = List.of("", "", " ", "ꭧ", "", "a");
        int[] out = Phonemizer.ipa(sparse, run -> "aꭧ").phonemize("x");
        assertArrayEquals(new int[] {5, 3}, out);
    }

    @Test
    void aDuplicateSymbolKeepsItsFirstId() {
        List<String> doubled = List.of("", " ", "'", "a", "'");
        int[] out = Phonemizer.ipa(doubled, run -> "'a'").phonemize("x");
        assertArrayEquals(new int[] {2, 3, 2}, out);
    }

    @Test
    void aMultiCodePointSymbolIsRefused() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Phonemizer.ipa(List.of("", " ", "ab"), run -> run));
        assertEquals("symbol 2 is not one Unicode code point: 'ab'", e.getMessage());
    }

    @Test
    void aTableWithoutASpaceIsRefused() {
        assertThrows(
                IllegalArgumentException.class,
                () -> Phonemizer.ipa(List.of("", "a", "b"), run -> run));
    }
}
