package com.qxotic.toknroll;

import com.qxotic.toknroll.impl.SpecialsImpl;
import java.util.Objects;
import java.util.Set;

/**
 * Precompiled special-token matcher for explicit special-aware encoding.
 *
 * <p>Special matching runs on raw input text before any tokenizer preprocessing, including a
 * configured normalizer and splitter. Non-special spans are then encoded via {@link
 * Tokenizer#encodeInto(CharSequence, int, int, IntSequence.Builder)}, so tokenizer preprocessing
 * still applies to ordinary text. Matched specials are injected as atomic token IDs resolved from
 * the provided {@link Vocabulary} at compile time.
 *
 * <p>A {@link Specials} instance must be used only with tokenizers that expose a compatible
 * vocabulary mapping for the compiled special tokens; otherwise behavior is undefined.
 *
 * <p>This API is intentionally separate from {@link Tokenizer} to keep core tokenization focused on
 * round-trip integrity for ordinary text and free from special-token policy. In TikToken terms,
 * {@link Tokenizer#encode(CharSequence)} corresponds to an ordinary-text path, while {@link
 * #encode(Tokenizer, CharSequence)} provides an explicit special-aware path.
 *
 * <p>Special-aware encoding is deterministic but policy-driven. It is not part of the ordinary text
 * round-trip guarantee.
 */
public interface Specials {

    /** Returns a singleton instance with no configured specials. */
    static Specials none() {
        return SpecialsImpl.none();
    }

    /**
     * Compiles a special token set against a vocabulary.
     *
     * <p>Validation rules:
     *
     * <ul>
     *   <li>{@code specials} may be empty (equivalent to {@link #none()}).
     *   <li>Each special token string must be non-null and non-empty.
     *   <li>Each special must exist in {@code vocabulary}.
     *   <li>No special may be a prefix of another special.
     * </ul>
     *
     * @throws NullPointerException if {@code vocabulary} or {@code specials} is null
     * @throws IllegalArgumentException if validation fails
     */
    static Specials compile(Vocabulary vocabulary, Set<String> specials) {
        return SpecialsImpl.compile(
                Objects.requireNonNull(vocabulary, "vocabulary"),
                Objects.requireNonNull(specials, "specials"));
    }

    /** Returns an immutable view of configured special token strings. */
    Set<String> tokens();

    /** Returns {@code true} if no special tokens are configured. */
    default boolean isEmpty() {
        return tokens().isEmpty();
    }

    /**
     * Encodes text with special-token injection and appends token IDs into {@code out}.
     *
     * <p>Matching happens on raw input text before tokenizer preprocessing (including normalizer
     * and splitter). Non-special spans are encoded through {@code tokenizer.encodeInto(...)}.
     *
     * @throws NullPointerException if {@code tokenizer}, {@code text}, or {@code out} is null
     */
    void encodeInto(Tokenizer tokenizer, CharSequence text, IntSequence.Builder out);

    /**
     * Encodes text with special-token injection.
     *
     * <p>This method is implemented in terms of {@link #encodeInto(Tokenizer, CharSequence,
     * IntSequence.Builder)}.
     *
     * @throws NullPointerException if {@code tokenizer} or {@code text} is null
     */
    default IntSequence encode(Tokenizer tokenizer, CharSequence text) {
        Objects.requireNonNull(tokenizer, "tokenizer");
        Objects.requireNonNull(text, "text");
        float ratio = Math.max(1.0e-6f, tokenizer.expectedTokensPerChar());
        int capacity = Math.max(8, (int) Math.ceil(text.length() * ratio * 1.15f) + 8);
        IntSequence.Builder out = IntSequence.newBuilder(capacity);
        encodeInto(tokenizer, text, out);
        return out.build();
    }
}
