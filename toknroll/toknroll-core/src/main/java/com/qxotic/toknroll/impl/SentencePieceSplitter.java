package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.Splitter;
import java.util.BitSet;
import java.util.Objects;

/**
 * Cuts SentencePiece text into words, before a space, wherever no token can span the cut: the words
 * merge into exactly the ids the whole text would, at a fraction of the cost and memory.
 *
 * <p>A SentencePiece model merges a piece whole, and every merge builds a vocabulary token. A token
 * spanning a cut before a space would hold the char before the cut right before a space, so a cut
 * is safe unless some token holds that char there. Such a char is glue. Marking more glue only cuts
 * less, so the split is exact for any vocabulary. A space is ' ' or U+2581, and the two are one:
 * the GGUF loader hands over its vocabulary spelled U+2581 and its text spelled ' ', the Hugging
 * Face loader both spelled U+2581. The examples spell ' '.
 *
 * <pre>{@code
 * tokens "a", " b"           "hello world"  ->  "hello" | " world"
 * tokens "a", " b"           "x> </"        ->  "x>" | " </"
 * tokens "> </" ('>' glue)   "x> </"        ->  "x> </"
 * tokens "a"                 "a   b"        ->  "a" | " " | " " | " b"
 * tokens "  " (' ' glue)     "a   b"        ->  "a" | "   b"
 * }</pre>
 *
 * <p>{@code isolateNewlines} makes each run of newlines a piece of its own, cut from the text on
 * both sides, as HF's {@code Isolated} split behavior and Gemma 4's GGUF regex {@code [^\n]+|[\n]+}
 * have it. Those cuts are required, not derived from the vocabulary: llama.cpp makes them whatever
 * tokens there are. Without it a newline is a char like any other, cut only where the space rule
 * cuts.
 *
 * <pre>{@code
 * tokens "a", " b"   "ab\n\ncd ef"   isolateNewlines   ->  "ab" | "\n\n" | "cd" | " ef"
 * tokens "a", " b"   "ab\n\ncd ef"   otherwise         ->  "ab\n\ncd" | " ef"
 * }</pre>
 */
final class SentencePieceSplitter implements Splitter {

    private final BitSet glue = new BitSet();
    private final boolean isolateNewlines;

    /**
     * @param tokens the vocabulary, a space spelled ' ' or U+2581; null entries are skipped
     * @param isolateNewlines whether every newline run is a piece of its own, cut on both sides
     */
    SentencePieceSplitter(String[] tokens, boolean isolateNewlines) {
        this.isolateNewlines = isolateNewlines;
        for (String token : tokens) {
            if (token == null) {
                continue;
            }
            for (int i = 1; i < token.length(); i++) {
                if (isSpace(token.charAt(i))) {
                    glue.set(spelled(token.charAt(i - 1)));
                }
            }
        }
    }

    private static boolean isSpace(char c) {
        return c == ' ' || c == '\u2581';
    }

    /** A space by one name, so a token's U+2581 is glue for the text's ' ' and back. */
    private static char spelled(char c) {
        return isSpace(c) ? ' ' : c;
    }

    @Override
    public void splitAll(CharSequence text, int start, int end, SplitConsumer consumer) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(consumer, "consumer");
        Objects.checkFromToIndex(start, end, text.length());
        int from = start;
        for (int i = start + 1; i < end; i++) {
            char c = text.charAt(i);
            char before = text.charAt(i - 1);
            if ((isolateNewlines && (c == '\n') != (before == '\n'))
                    || (isSpace(c) && !glue.get(spelled(before)))) {
                consumer.accept(text, from, i);
                from = i;
            }
        }
        if (from < end) {
            consumer.accept(text, from, end);
        }
    }
}
