// espeak-ng subprocess phonemizer. Finds the binary on PATH at construction time.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.*;
import java.util.concurrent.TimeUnit;

final class EspeakPhonemizer implements Phonemizer {
    private final String binary;

    private EspeakPhonemizer(String binary) {
        this.binary = binary;
    }

    static EspeakPhonemizer tryCreate() {
        for (String name : new String[] {"espeak-ng", "espeak"}) {
            try {
                var p = new ProcessBuilder(name, "--version").redirectErrorStream(true).start();
                if (p.waitFor(2, TimeUnit.SECONDS) && p.exitValue() == 0)
                    return new EspeakPhonemizer(name);
            } catch (Exception ignored) {
            }
        }
        return null;
    }

    /**
     * Convert English text to blank-interspersed model tokens. The caller should pre-normalize via
     * {@link TextNormalizer} (word overrides, numbers, dates, etc.). Punctuation is preserved by
     * tokenizing word-by-word — matching the Python {@code preserve_punctuation=True} mode.
     */
    @Override
    public int[] phonemize(String text) throws Exception {
        // Phonemize word-by-word so punctuation is preserved.
        // espeak strips punctuation from input, so we extract word + trailing punct,
        // phonemize each word, then reattach the punctuation.
        StringBuilder sb = new StringBuilder();
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            // Find trailing punctuation: scan backwards until letter/digit or
            // a contraction apostrophe (don't, can't, it's).
            int punctStart = token.length();
            while (punctStart > 0) {
                int cp = token.codePointAt(punctStart - 1);
                if (Character.isLetter(cp) || Character.isDigit(cp)) break;
                if (cp == '\''
                        && punctStart > 1
                        && Character.isLetter(token.codePointBefore(punctStart - 1))) break;
                punctStart--;
            }
            String word = token.substring(0, punctStart);
            String punct = token.substring(punctStart);

            if (!word.isEmpty()) sb.append(phonemizeWord(word));
            if (!punct.isEmpty()) sb.append(punct);
            sb.append(' ');
        }
        return Symbols.toTokens(sb.toString().replaceAll("\\s+", " ").trim());
    }

    private String phonemizeWord(String word) throws Exception {
        var p =
                new ProcessBuilder(binary, "--ipa", "-q", "-v", "en-us", word)
                        .redirectErrorStream(true)
                        .start();
        String out;
        try (var r = new BufferedReader(new InputStreamReader(p.getInputStream()))) {
            out = r.lines().reduce("", (a, b) -> a + b);
        }
        p.waitFor(5, TimeUnit.SECONDS);
        return out.replace("_", "").replaceAll("\\s+", " ").trim();
    }
}
