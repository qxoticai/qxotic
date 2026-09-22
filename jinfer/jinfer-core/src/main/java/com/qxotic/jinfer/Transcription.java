package com.qxotic.jinfer;

import java.time.Duration;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A speech-to-text result: the transcript, and the emitted tokens with the audio spans they were
 * aligned to. Token times are offsets from the start of the input; {@link #words()} groups the
 * tokens into words.
 */
public record Transcription(String text, List<Token> tokens) {

    private static final Transcription EMPTY = new Transcription("", List.of());

    public Transcription {
        Objects.requireNonNull(text, "text");
        tokens = List.copyOf(tokens);
    }

    /** No text and no tokens; one shared instance. */
    public static Transcription empty() {
        return EMPTY;
    }

    /**
     * Tokens grouped into words: a token whose text opens with a space starts a new word. A word
     * spans its tokens' time span and carries their weakest confidence: a word is only as certain
     * as its least certain piece.
     */
    public List<Word> words() {
        List<Word> words = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        Duration start = Duration.ZERO, end = Duration.ZERO;
        double confidence = 1;
        for (Token token : tokens) {
            if (token.text().startsWith(" ") && !current.isEmpty()) {
                words.add(new Word(current.toString().strip(), start, end, confidence));
                current.setLength(0);
                confidence = 1;
            }
            if (current.isEmpty()) start = token.start();
            current.append(token.text());
            end = token.end();
            confidence = Math.min(confidence, token.confidence());
        }
        if (!current.isEmpty())
            words.add(new Word(current.toString().strip(), start, end, confidence));
        return words;
    }

    /**
     * One word of the transcript: its {@code [start, end)} span and its weakest token's confidence.
     */
    public record Word(String text, Duration start, Duration end, double confidence) {
        public Word {
            Objects.requireNonNull(text, "text");
            requireSpan(start, end);
        }
    }

    /**
     * One emitted token: its decoded text, its {@code [start, end)} span, and the decoder's
     * confidence in it. Confidence is the maximum probability over the token vocabulary, rescaled
     * as {@code (N*p - 1)/(N - 1)}: 0 is a uniform guess, 1 a fully peaked one.
     */
    public record Token(String text, Duration start, Duration end, double confidence) {
        public Token {
            Objects.requireNonNull(text, "text");
            requireSpan(start, end);
            if (!(confidence >= 0 && confidence <= 1))
                throw new IllegalArgumentException("confidence out of [0,1]: " + confidence);
        }
    }

    private static void requireSpan(Duration start, Duration end) {
        if (start.isNegative() || end.compareTo(start) < 0)
            throw new IllegalArgumentException("invalid span " + start + ".." + end);
    }
}
