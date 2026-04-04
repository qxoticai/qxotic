package com.qxotic.toknroll;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Tokenizer contract with four core operations: {@link #encodeInto(CharSequence, int, int,
 * IntSequence.Builder)}, {@link #decodeBytesInto(IntSequence, int, ByteBuffer)}, {@link
 * #countTokens(CharSequence)}, and {@link #countBytes(IntSequence)}.
 *
 * <p>All other methods are convenience wrappers built on top of these operations.
 */
public interface Tokenizer {

    Vocabulary vocabulary();

    /**
     * Encodes {@code text[startInclusive, endExclusive)} and appends token IDs to {@code out}.
     *
     * @throws IndexOutOfBoundsException if the slice range is invalid
     */
    void encodeInto(
            CharSequence text, int startInclusive, int endExclusive, IntSequence.Builder out);

    /**
     * Writes decoded bytes for tokens starting at {@code tokenStartIndex} into {@code out}.
     *
     * <p>Returns consumed token count. Returns {@code 0} only when {@code tokenStartIndex ==
     * tokens.length()}. Implementations must not write partial token bytes.
     *
     * <p>If the next token cannot fit and no token is consumed, throws {@link
     * IllegalArgumentException}. If some tokens are consumed, implementations may return early.
     *
     * @throws IndexOutOfBoundsException if {@code tokenStartIndex} is outside {@code [0,
     *     tokens.length()]}
     * @throws IllegalArgumentException if the next token does not fit in {@code out.remaining()}
     *     and no token was consumed
     */
    int decodeBytesInto(IntSequence tokens, int tokenStartIndex, ByteBuffer out);

    /**
     * Expected tokens-per-character hint used only for internal preallocation.
     *
     * <p>This value does not affect correctness.
     */
    default float expectedTokensPerChar() {
        return 0.5f;
    }

    /** Returns token count for encoding {@code text}. Must match {@code encode(text).length()}. */
    default int countTokens(CharSequence text) {
        Objects.requireNonNull(text, "text");
        IntSequence.Builder out = IntSequence.newBuilder(estimateInitialTokenCapacity(text));
        encodeInto(text, out);
        return out.size();
    }

    /**
     * Returns decoded byte count for {@code tokens}. Must match {@code decodeBytes(tokens).length}.
     */
    default int countBytes(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int tokenCount = tokens.length();
        if (tokenCount == 0) {
            return 0;
        }

        byte[] scratch = new byte[256];
        ByteBuffer out = ByteBuffer.wrap(scratch);
        int tokenIndex = 0;
        int totalBytes = 0;
        while (tokenIndex < tokenCount) {
            out.clear();
            int consumedTokens = decodeBytesInto(tokens, tokenIndex, out);
            if (consumedTokens <= 0) {
                throw new IllegalStateException(
                        "decodeBytesInto made no progress at token index " + tokenIndex);
            }
            tokenIndex += consumedTokens;
            totalBytes += out.position();
        }
        return totalBytes;
    }

    /** Encodes full text and appends token IDs to {@code out}. */
    default void encodeInto(CharSequence text, IntSequence.Builder out) {
        Objects.requireNonNull(text, "text");
        encodeInto(text, 0, text.length(), out);
    }

    /** Encodes text into an immutable token sequence. */
    default IntSequence encode(CharSequence text) {
        Objects.requireNonNull(text, "text");
        IntSequence.Builder out = IntSequence.newBuilder(estimateInitialTokenCapacity(text));
        encodeInto(text, out);
        return out.build();
    }

    private int estimateInitialTokenCapacity(CharSequence text) {
        int charCount = text.length();
        float ratio = Math.max(1.0e-6f, expectedTokensPerChar());
        int predicted = (int) Math.ceil(charCount * ratio * 1.15f) + 8;
        return Math.max(8, predicted);
    }

    /** Encodes text and returns token IDs as an array. */
    default int[] encodeToArray(CharSequence text) {
        return encode(text).toArray();
    }

    /** Decodes tokens into a byte array sized via {@link #countBytes(IntSequence)}. */
    default byte[] decodeBytes(IntSequence tokens) {
        Objects.requireNonNull(tokens, "tokens");
        int tokenCount = tokens.length();
        int totalBytes = countBytes(tokens);
        byte[] out = new byte[totalBytes];
        ByteBuffer buffer = ByteBuffer.wrap(out);

        int tokenIndex = 0;
        while (tokenIndex < tokenCount) {
            int consumed = decodeBytesInto(tokens, tokenIndex, buffer);
            if (consumed <= 0) {
                throw new IllegalStateException(
                        "decodeBytesInto made no progress at token index " + tokenIndex);
            }
            tokenIndex += consumed;
        }
        return out;
    }

    /** Counts decoded bytes for a token-ID array. */
    default int countBytes(int[] tokens) {
        return countBytes(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }

    /** Decodes token IDs from an int array into raw bytes. */
    default byte[] decodeBytes(int[] tokens) {
        return decodeBytes(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }

    /** Decodes tokens into UTF-8 text. */
    default String decode(IntSequence tokens) {
        return new String(
                decodeBytes(Objects.requireNonNull(tokens, "tokens")), StandardCharsets.UTF_8);
    }

    /** Decodes token IDs from an int array into UTF-8 text. */
    default String decode(int[] tokens) {
        return decode(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }
}
