package com.qxotic.tokenizers.advanced;

import java.text.Normalizer.Form;
import java.util.Locale;
import java.util.Objects;
import java.util.function.Function;

@FunctionalInterface
public interface Normalizer extends Function<CharSequence, CharSequence> {

    /** Strict default: no text transformation. */
    Normalizer IDENTITY = text -> text;

    /** Strict default: no text transformation. */
    static Normalizer identity() {
        return IDENTITY;
    }

    /** Lossy transform. Opt-in only. */
    static Normalizer lowercase() {
        return text -> text.toString().toLowerCase(Locale.ROOT);
    }

    /** Composes normalizers in order. */
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

    /** Unicode canonicalization. */
    static Normalizer unicode(Form form) {
        Objects.requireNonNull(form, "form");
        return text -> java.text.Normalizer.normalize(text, form);
    }
}
