package com.qxotic.toknroll;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.util.Objects;

/**
 * Practical tokenizer contract with four core operations: {@link #encodeInto(CharSequence, int,
 * int, IntSequence.Builder)}, {@link #decodeBytesInto(IntSequence, int, ByteBuffer)}, {@link
 * #countTokens(CharSequence, int, int)}, and {@link #countBytes(IntSequence)}.
 *
 * <p>All other methods are convenience wrappers built on top of these operations.
 *
 * <p>This interface is intentionally flexible: implementations may apply model-specific policy,
 * normalization, special-token handling, or lossy text paths. As a result, text-level reversibility
 * is not guaranteed in general (see {@link TokenizationModel} for the strict, reversible model
 * contract).
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
     * Writes decoded bytes for tokens {@code [tokenStartIndex, ...)} into {@code out}, stopping
     * when the buffer is full or the sequence is exhausted.
     *
     * <p>Returns the number of tokens consumed (always ≥ 1) unless {@code tokenStartIndex ==
     * tokens.length()}, in which case returns {@code 0}. Implementations never write partial token
     * bytes: a token is either written in full or skipped entirely.
     *
     * <p>Callers that need to decode the full sequence should loop, advancing {@code
     * tokenStartIndex} by the return value each iteration until {@code tokenStartIndex ==
     * tokens.length()}.
     *
     * @throws IndexOutOfBoundsException if {@code tokenStartIndex} is outside {@code [0,
     *     tokens.length()]}
     * @throws IllegalArgumentException if the first token at {@code tokenStartIndex} does not fit
     *     in {@code out.remaining()} (i.e., no progress can be made)
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

    /**
     * Returns token count for encoding {@code text}.
     *
     * <p>Equivalent to {@code countTokens(text, 0, text.length())}.
     */
    default int countTokens(CharSequence text) {
        Objects.requireNonNull(text, "text");
        return countTokens(text, 0, text.length());
    }

    /**
     * Returns token count for encoding {@code text[startInclusive, endExclusive)}.
     *
     * <p>Must equal {@code encode(text.subSequence(startInclusive, endExclusive)).length()}.
     *
     * @throws IndexOutOfBoundsException if the slice range is invalid
     */
    int countTokens(CharSequence text, int startInclusive, int endExclusive);

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

    /**
     * Encodes text into an immutable token sequence using the ordinary (non-special-aware) path.
     */
    default IntSequence encode(CharSequence text) {
        Objects.requireNonNull(text, "text");
        IntSequence.Builder out =
                IntSequence.newBuilder(
                        estimateTokenCapacity(text.length(), expectedTokensPerChar()));
        encodeInto(text, out);
        return out.build();
    }

    /**
     * Pre-allocates a reasonable initial capacity for {@link IntSequence.Builder} given character
     * count and the model's expected tokens-to-char ratio.
     */
    static int estimateTokenCapacity(int charCount, float expectedTokensPerChar) {
        float ratio = Math.max(1.0e-6f, expectedTokensPerChar);
        double estimated = Math.ceil(charCount * (double) ratio * 1.15d) + 8d;
        if (estimated > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Estimated token capacity exceeds int range: " + estimated);
        }
        int predicted = (int) estimated;
        return Math.max(8, predicted);
    }

    /**
     * Encodes {@code text} and returns token IDs as a plain {@code int[]}.
     *
     * <p>Equivalent to {@code encode(text).toArray()}. For performance-sensitive paths, prefer
     * {@link #encodeInto} to avoid the intermediate allocation.
     */
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

    /**
     * Counts decoded bytes for a token-ID array.
     *
     * @param tokens token IDs to decode
     * @return total decoded byte count
     */
    default int countBytes(int[] tokens) {
        return countBytes(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }

    /**
     * Decodes token IDs from an int array into raw bytes.
     *
     * @param tokens token IDs to decode
     * @return decoded raw bytes
     */
    default byte[] decodeBytes(int[] tokens) {
        return decodeBytes(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }

    /**
     * Decodes tokens into UTF-8 text.
     *
     * <p>This text decode is convenient but may be lossy for arbitrary token sequences. Use {@link
     * #decodeBytes(IntSequence)} when full byte fidelity is required.
     */
    default String decode(IntSequence tokens) {
        return new String(
                decodeBytes(Objects.requireNonNull(tokens, "tokens")), StandardCharsets.UTF_8);
    }

    /**
     * Decodes token IDs from an int array into UTF-8 text.
     *
     * @param tokens token IDs to decode
     * @return decoded UTF-8 text
     */
    default String decode(int[] tokens) {
        return decode(IntSequence.wrap(Objects.requireNonNull(tokens, "tokens")));
    }
}
