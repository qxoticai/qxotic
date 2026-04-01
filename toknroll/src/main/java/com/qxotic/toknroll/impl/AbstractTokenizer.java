package com.qxotic.toknroll.impl;

import com.qxotic.toknroll.*;
import com.qxotic.toknroll.advanced.Normalizer;
import com.qxotic.toknroll.advanced.Splitter;
import java.util.Objects;

/**
 * Abstract base implementation of the {@link Tokenizer} interface that provides a configurable
 * tokenization pipeline with normalization and splitting steps.
 *
 * <p>The tokenization process follows these steps:
 *
 * <ol>
 *   <li>Text normalization using a configured {@link Normalizer}
 *   <li>Text splitting into chunks using a configured {@link Splitter}
 *   <li>Token encoding for each chunk using the implementation-specific algorithm
 * </ol>
 *
 * <p>Implementations need to provide:
 *
 * <ul>
 *   <li>The core encoding logic via {@link #encodeImpl(CharSequence)}
 *   <li>The decoding logic via {@link #decodeBytesInto(IntSequence, int, java.nio.ByteBuffer)}
 * </ul>
 *
 * <p>Example usage:
 *
 * <pre>{@code
 * class MyTokenizer extends AbstractTokenizer {
 *     public MyTokenizer(Vocabulary vocab) {
 *         super(vocab, myNormalizer, mySplitter);
 *     }
 *
 *     @Override
 *     protected IntSequence encodeImpl(CharSequence text) {
 *         // Implementation-specific encoding logic
 *     }
 *
 *     @Override
 *     public int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out) {
 *         // Implementation-specific decode streaming logic
 *     }
 * }
 * }</pre>
 */
public abstract class AbstractTokenizer implements Tokenizer {
    /** The vocabulary used for token lookup. */
    protected final Vocabulary vocabulary;

    /** The normalizer used for text preprocessing. */
    protected final Normalizer normalizer;

    /** The splitter used to break text into chunks. */
    protected final Splitter splitter;

    /**
     * Creates a new tokenizer with custom normalization and splitting behavior.
     *
     * @param vocabulary the vocabulary for token lookup
     * @param normalizer the normalizer for text preprocessing
     * @param splitter the splitter for breaking text into chunks
     * @throws NullPointerException if any parameter is null
     */
    protected AbstractTokenizer(Vocabulary vocabulary, Normalizer normalizer, Splitter splitter) {
        this.vocabulary = Objects.requireNonNull(vocabulary, "vocabulary");
        this.normalizer = Objects.requireNonNull(normalizer, "normalizer");
        this.splitter = Objects.requireNonNull(splitter, "splitter");
    }

    /**
     * Creates a new tokenizer with identity normalization and splitting.
     *
     * @param vocabulary the vocabulary for token lookup
     * @throws NullPointerException if vocabulary is null
     */
    protected AbstractTokenizer(Vocabulary vocabulary) {
        this(vocabulary, Normalizer.identity(), Splitter.identity());
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

        CharSequence normalizedPart =
                normalizer.apply(text.subSequence(startInclusive, endExclusive));
        splitter.splitAll(
                normalizedPart,
                0,
                normalizedPart.length(),
                (source, chunkStart, chunkEnd) ->
                        encodeImplInto(source.subSequence(chunkStart, chunkEnd), out));
    }

    /**
     * Implements the core encoding logic for a chunk of text. This method is called after
     * normalization and splitting, and should implement the actual token generation algorithm.
     *
     * @param text the chunk of text to encode
     * @return sequence of token IDs representing the text
     * @throws IllegalArgumentException if the text cannot be encoded
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
    public int countTokens(CharSequence text) {
        return encode(text).length();
    }
}
