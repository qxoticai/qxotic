package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.*;
import java.util.Objects;

/**
 * Abstract base implementation of the {@link Tokenizer} interface.
 *
 * <p>Provides vocabulary management and a default encoding pipeline. Subclasses implement the core
 * chunk-level encoding logic via {@link #encodeImplInto(CharSequence, IntSequence.Builder)}.
 * Normalization and splitting are handled externally by {@link
 * com.qxotic.toknroll.TokenizationPipeline}.
 *
 * <p>Implementations need to provide:
 *
 * <ul>
 *   <li>The core encoding logic via {@link #encodeImpl(CharSequence)}
 *   <li>The decoding logic via {@link #decodeBytesInto(IntSequence, int, java.nio.ByteBuffer)}
 * </ul>
 */
public abstract class AbstractTokenizer implements Tokenizer {

    /** The vocabulary used for token lookup. */
    protected final Vocabulary vocabulary;

    /**
     * Creates a new tokenizer with the given vocabulary.
     *
     * @param vocabulary the vocabulary for token lookup
     * @throws NullPointerException if vocabulary is null
     */
    protected AbstractTokenizer(Vocabulary vocabulary) {
        this.vocabulary = Objects.requireNonNull(vocabulary, "vocabulary");
    }

    @Override
    public Vocabulary vocabulary() {
        return this.vocabulary;
    }

    @Override
    public void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        Objects.requireNonNull(out, "out");
        if (startInclusive < 0 || endExclusive < startInclusive || endExclusive > text.length()) {
            throw new IndexOutOfBoundsException(
                    "Invalid range ["
                            + startInclusive
                            + ", "
                            + endExclusive
                            + ") for text length "
                            + text.length());
        }
        encodeImplInto(text.subSequence(startInclusive, endExclusive), out);
    }

    /**
     * Implements the core encoding logic for a chunk of text. This method is called with a
     * pre-normalized, pre-split chunk and should implement the actual token generation algorithm.
     *
     * @param text the chunk of text to encode
     * @return sequence of token IDs representing the text
     */
    protected abstract IntSequence encodeImpl(CharSequence text);

    /**
     * Appends encoded token IDs for the given chunk into {@code out}.
     *
     * <p>Default implementation delegates to {@link #encodeImpl(CharSequence)} and appends the
     * resulting sequence. Implementations can override to avoid intermediate allocations.
     */
    protected void encodeImplInto(CharSequence text, IntSequence.Builder out) {
        out.addAll(encodeImpl(text));
    }

    @Override
    public int countTokens(CharSequence text, int startInclusive, int endExclusive) {
        Objects.requireNonNull(text, "text");
        int charCount = endExclusive - startInclusive;
        float ratio = Math.max(1.0e-6f, expectedTokensPerChar());
        IntSequence.Builder out =
                IntSequence.newBuilder(Math.max(8, (int) Math.ceil(charCount * ratio * 1.15f) + 8));
        encodeInto(text, startInclusive, endExclusive, out);
        return out.size();
    }
}
