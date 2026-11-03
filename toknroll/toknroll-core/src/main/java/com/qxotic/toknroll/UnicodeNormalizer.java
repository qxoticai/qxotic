package com.qxotic.toknroll;

import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer.Form;
import java.util.Arrays;

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
     * {@code utf8[from, to)} decoded, normalized and encoded again, or null when that gives the
     * same bytes.
     *
     * <p>No form changes ASCII or combines it with what comes before it, and decoding
     * resynchronizes at it, so each run of non-ASCII bytes normalizes on its own, together with the
     * ASCII byte before it, which may compose with it. Only those runs are decoded.
     */
    byte[] normalizeUtf8(byte[] utf8, int from, int to) {
        ByteArrayOutputStream out = null;
        int copied = from; // utf8[copied, start) is unchanged, and not in out yet
        for (int i = from; i < to; ) {
            if (utf8[i] >= 0) {
                i++;
                continue;
            }
            int start = i > from ? i - 1 : i;
            while (i < to && utf8[i] < 0) {
                i++;
            }
            String run = new String(utf8, start, i - start, StandardCharsets.UTF_8);
            byte[] normalized =
                    java.text.Normalizer.normalize(run, form).getBytes(StandardCharsets.UTF_8);
            if (!Arrays.equals(normalized, 0, normalized.length, utf8, start, i)) {
                if (out == null) {
                    out = new ByteArrayOutputStream(to - from + normalized.length);
                }
                out.write(utf8, copied, start - copied);
                out.write(normalized, 0, normalized.length);
                copied = i;
            }
        }
        if (out == null) {
            return null;
        }
        out.write(utf8, copied, to - copied);
        return out.toByteArray();
    }
}
