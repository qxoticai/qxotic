package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.ImplAccessor;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Objects;

/**
 * A tokenizer's special tokens, matched whole for explicit special-aware encoding.
 *
 * <p>The special tokens are the ones a tokenizer's vocabulary types {@link
 * StandardTokenType#CONTROL CONTROL}, {@link StandardTokenType#USER_DEFINED USER_DEFINED} or {@link
 * StandardTokenType#UNKNOWN UNKNOWN}, each known by the text it decodes to. Neither the vocabulary
 * nor the tokenizer matches them: {@link Tokenizer#encode(CharSequence)} treats every character as
 * text, and only {@link #encode(Tokenizer, CharSequence)} turns a special token's text into its id.
 *
 * <p>Matching runs on the raw input, before any normalizer or splitter. At each position the
 * longest special token that starts there wins, and the scan continues after it, so where one
 * special token begins another (Laguna's {@code 〈|} and {@code 〈|EOS|〉}) both stay reachable. This
 * is Hugging Face's rule; llama.cpp gives the same ids wherever special tokens do not overlap at
 * different positions. The text between special tokens is encoded with {@link
 * Tokenizer#encodeInto(CharSequence, int, int, IntSequence.Builder)}, as ranges of the one input.
 *
 * <p>Use it for text you wrote, such as a rendered chat template: a user's text containing a
 * special token's spelling would become that token. Encode user text with {@link
 * Tokenizer#encode(CharSequence)} and splice the special tokens in by id, from {@link #tokens()}.
 *
 * <p>In Tiktoken terms, {@link Tokenizer#encode(CharSequence)} is {@code encode_ordinary} and
 * {@link #encode(Tokenizer, CharSequence)} is {@code encode(..., allowed_special="all")}; in
 * Hugging Face's, they are the tokenizer without its added tokens and {@code encode}; in
 * llama.cpp's, no added token matched and {@code parse_special}.
 */
public interface Specials {

    /** No special tokens: encoding with it is {@link Tokenizer#encode(CharSequence)}. */
    static Specials none() {
        return ImplAccessor.specialsNone();
    }

    /**
     * The special tokens of {@code tokenizer}: the tokens its vocabulary types CONTROL,
     * USER_DEFINED or UNKNOWN, each by the text {@code tokenizer} decodes it to after another
     * token, so a space a decoder trims from the start of the text stays. A token that decodes to
     * no text is left out; where two tokens decode to the same text, the lower id wins. Build once
     * and reuse: it scans the whole vocabulary.
     *
     * @param tokenizer the tokenizer whose special tokens to match
     * @return the special tokens, {@link #none()} when the vocabulary types none
     * @throws NullPointerException if {@code tokenizer} is null
     */
    static Specials of(Tokenizer tokenizer) {
        return ImplAccessor.specialsOf(Objects.requireNonNull(tokenizer, "tokenizer"));
    }

    /**
     * The special tokens by their text, unmodifiable: a special token's id is {@code
     * tokens().get(text)}.
     *
     * @return text to id
     */
    Map<String, Integer> tokens();

    /**
     * Returns {@code true} if there are no special tokens.
     *
     * @return {@code true} if {@link #tokens()} is empty
     */
    default boolean isEmpty() {
        return tokens().isEmpty();
    }

    /**
     * Encodes {@code text}, special tokens matched whole, and appends the ids to {@code out}.
     *
     * @param tokenizer the tokenizer for the text between special tokens, the one these special
     *     tokens came from
     * @param text input text
     * @param out builder to append token IDs to
     * @throws NullPointerException if {@code tokenizer}, {@code text}, or {@code out} is null
     */
    void encodeInto(Tokenizer tokenizer, CharSequence text, IntSequence.Builder out);

    /**
     * Encodes UTF-8 text with special-token injection and appends token IDs into {@code out}: for
     * well-formed UTF-8, the same IDs as {@link #encodeInto(Tokenizer, CharSequence,
     * IntSequence.Builder)} gives the decoded text. The special tokens {@link #of} finds are
     * matched on the bytes, and the text between them is encoded through {@code
     * tokenizer.encodeUtf8Into(...)}, so it is never decoded; this default implementation decodes
     * the range and calls {@code encodeInto}.
     *
     * @param tokenizer the tokenizer for the text between special tokens, the one these special
     *     tokens came from
     * @param utf8 the text, as UTF-8
     * @param startInclusive start of the range to encode (inclusive)
     * @param endExclusive end of the range to encode (exclusive)
     * @param out builder to append token IDs to
     * @throws NullPointerException if {@code tokenizer}, {@code utf8}, or {@code out} is null
     * @throws IndexOutOfBoundsException if the range is outside {@code utf8}
     */
    default void encodeUtf8Into(
            Tokenizer tokenizer,
            byte[] utf8,
            int startInclusive,
            int endExclusive,
            IntSequence.Builder out) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(utf8, "utf8");
        Objects.requireNonNull(out, "out");
        Objects.checkFromToIndex(startInclusive, endExclusive, utf8.length);
        encodeInto(
                tokenizer,
                new String(
                        utf8,
                        startInclusive,
                        endExclusive - startInclusive,
                        StandardCharsets.UTF_8),
                out);
    }

    /**
     * Encodes {@code text}, special tokens matched whole.
     *
     * @param tokenizer the tokenizer for the text between special tokens, the one these special
     *     tokens came from
     * @param text input text
     * @return immutable token sequence
     * @throws NullPointerException if {@code tokenizer} or {@code text} is null
     */
    default IntSequence encode(Tokenizer tokenizer, CharSequence text) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "text");
        float ratio = Math.max(1.0e-6f, tokenizer.expectedTokensPerChar());
        double estimated = Math.ceil(text.length() * (double) ratio * 1.15d) + 8d;
        if (estimated > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Estimated token capacity exceeds int range: " + estimated);
        }
        int capacity = Math.max(8, (int) estimated);
        IntSequence.Builder out = IntSequence.newBuilder(capacity);
        encodeInto(tokenizer, text, out);
        return out.build();
    }
}
