// espeak-ng subprocess phonemizer. Finds the binary on PATH at construction time.
package com.qxotic.jinfer.models.inflect2.frontend;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.concurrent.TimeUnit;

final class EspeakPhonemizer implements Phonemizer {

    private static final long TIMEOUT_SECONDS = 5;

    private final String binary;

    private EspeakPhonemizer(String binary) {
        this.binary = binary;
    }

    static EspeakPhonemizer tryCreate() {
        for (String name : new String[] {"espeak-ng", "espeak"}) {
            try {
                var probe = new ProcessBuilder(name, "--version").redirectErrorStream(true).start();
                if (probe.waitFor(2, TimeUnit.SECONDS) && probe.exitValue() == 0)
                    return new EspeakPhonemizer(name);
            } catch (IOException ignored) {
                // not installed under this name
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return null;
            }
        }
        return null;
    }

    /**
     * Convert English text to blank-interspersed model tokens. The caller should pre-normalize via
     * {@link TextNormalizer}. espeak drops punctuation, so words are phonemized one at a time and
     * their trailing punctuation reattached — the Python frontend's {@code
     * preserve_punctuation=True} mode.
     */
    @Override
    public int[] phonemize(String text) throws IOException {
        var out = new StringBuilder();
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            int end = wordEnd(token);
            if (end > 0) out.append(phonemizeWord(token.substring(0, end)));
            out.append(token, end, token.length()).append(' ');
        }
        return Symbols.toTokens(out.toString().replaceAll("\\s+", " ").trim());
    }

    /**
     * Where the word ends and its trailing punctuation begins. An apostrophe between letters is
     * part of the word ("don't"), not punctuation.
     */
    /** Where the word ends and its trailing punctuation begins; shared with the lexicon path. */
    static int wordEnd(String token) {
        int end = token.length();
        while (end > 0) {
            int codePoint = token.codePointAt(end - 1);
            if (Character.isLetterOrDigit(codePoint)) break;
            if (codePoint == '\'' && end > 1 && Character.isLetter(token.codePointBefore(end - 1)))
                break;
            end--;
        }
        return end;
    }

    private String phonemizeWord(String word) throws IOException {
        Process espeak =
                new ProcessBuilder(binary, "--ipa", "-q", "-v", "en-us", word)
                        .redirectErrorStream(true)
                        .start();
        String ipa;
        try (var reader = new BufferedReader(new InputStreamReader(espeak.getInputStream()))) {
            ipa = String.join("", reader.lines().toList());
        }
        try {
            if (!espeak.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                espeak.destroyForcibly();
                throw new IOException(binary + " timed out on: " + word);
            }
        } catch (InterruptedException e) {
            espeak.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException(binary + " interrupted", e);
        }
        return ipa.replace("_", "").replaceAll("\\s+", " ").trim();
    }
}
