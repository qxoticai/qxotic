package com.qxotic.toknroll;

import java.text.Normalizer.Form;
import java.util.Locale;
import java.util.Objects;

/**
 * Preprocessing transformation applied to input text before tokenization.
 *
 * <p>Implementations must be stateless and thread-safe. The default normalizer ({@link
 * #identity()}) performs no transformation, which preserves round-trip fidelity. Lossy normalizers
 * such as {@link #lowercase()} must be explicitly opted into.
 */
@FunctionalInterface
public interface Normalizer {

    /** Identity normalizer; returns text unchanged. Equivalent to {@link #identity()}. */
    Normalizer IDENTITY = text -> text;

    /**
     * Applies this normalizer to the given text.
     *
     * @param text input text
     * @return transformed text
     */
    CharSequence apply(CharSequence text);

    /**
     * Strict default: no text transformation.
     *
     * @return identity normalizer
     */
    static Normalizer identity() {
        return IDENTITY;
    }

    /**
     * Lossy transform. Opt-in only.
     *
     * @return lowercase normalizer
     */
    static Normalizer lowercase() {
        return text -> text.toString().toLowerCase(Locale.ROOT);
    }

    /**
     * Composes normalizers in order.
     *
     * @param normalizers normalizers to chain
     * @return composed normalizer (identity if empty)
     */
    static Normalizer sequence(Normalizer... normalizers) {
        Objects.requireNonNull(normalizers, "normalizers");
        if (normalizers.length == 0) {
            return identity();
        }
        return text -> {
            CharSequence current = text;
            for (Normalizer normalizer : normalizers) {
                current = Objects.requireNonNull(normalizer, "normalizer").apply(current);
            }
            return current;
        };
    }

    /**
     * Unicode canonicalization.
     *
     * @param form Unicode normalization form
     * @return unicode normalizer
     */
    static Normalizer unicode(Form form) {
        Objects.requireNonNull(form, "form");
        return text -> java.text.Normalizer.normalize(text, form);
    }
}
