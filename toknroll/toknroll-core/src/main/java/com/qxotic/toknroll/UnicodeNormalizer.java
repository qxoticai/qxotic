package com.qxotic.toknroll;

import java.nio.charset.StandardCharsets;
import java.text.Normalizer.Form;

/** {@link Normalizer#unicode}: a Unicode normalization form, which leaves ASCII as it is. */
final class UnicodeNormalizer implements Normalizer {
    private final Form form;

    UnicodeNormalizer(Form form) {
        this.form = form;
    }

    @Override
    public CharSequence apply(CharSequence text) {
        return java.text.Normalizer.normalize(text, form);
    }

    /**
     * Whether well-formed {@code utf8[from, to)} is already in this form, so its bytes encode as
     * they are. Ill-formed bytes answer false: they decode to U+FFFD first.
     */
    boolean leaves(byte[] utf8, int from, int to) {
        int i = from;
        while (i < to && utf8[i] >= 0) {
            i++;
        }
        if (i == to) {
            return true;
        }
        String text = new String(utf8, from, to - from, StandardCharsets.UTF_8);
        // ponytail: a U+FFFD spelled in the text takes the decoding path too, which gives the same
        return text.indexOf('\uFFFD') < 0 && java.text.Normalizer.isNormalized(text, form);
    }
}
