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
     * {@link TextNormalizer}. espeak drops punctuation, so the text is phonemized in RUNS between
     * punctuation marks and the marks are reattached - the Python frontend's {@code
     * preserve_punctuation=True} mode, which likewise phonemizes whole spans rather than words.
     *
     * <p>A run, not a word, because espeak reads context: per word it stamps a primary stress on
     * every function word ({@code ðˈə}, {@code kˈæn}, {@code bˈiː}) and can never flap across a
     * boundary ({@code ˈɪt ˈɪz} vs {@code ɪɾ ɪz}). Measured on a 36-word paragraph: 6 subprocesses
     * instead of 36, and 8% less audio for the same text - the unstressed syllables shortening, not
     * a truncation. Warm synthesis of that paragraph went 0.301 s to 0.221 s (40.7x to 50.8x
     * realtime), against the lexicon path's 0.202 s - this front end is no longer the cliff it was.
     */
    @Override
    public int[] phonemize(String text) throws IOException {
        var out = new StringBuilder();
        var run = new StringBuilder();
        for (String token : text.split("\\s+")) {
            if (token.isEmpty()) continue;
            int end = wordEnd(token);
            if (end > 0) run.append(token, 0, end).append(' ');
            if (end < token.length()) {
                flush(run, out);
                out.append(token, end, token.length()).append(' ');
            }
        }
        flush(run, out);
        return Symbols.toTokens(out.toString().replaceAll("\\s+", " ").trim());
    }

    /** One espeak call for a whole punctuation-free run; espeak separates the words itself. */
    private void flush(StringBuilder run, StringBuilder out) throws IOException {
        if (run.isEmpty()) return;
        out.append(phonemizeRun(run.toString().trim())).append(' ');
        run.setLength(0);
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

    private String phonemizeRun(String words) throws IOException {
        Process espeak =
                new ProcessBuilder(binary, "--ipa", "-q", "-v", "en-us", words)
                        .redirectErrorStream(true)
                        .start();
        String ipa;
        try (var reader = new BufferedReader(new InputStreamReader(espeak.getInputStream()))) {
            // a run can span several output lines (espeak breaks at clause boundaries): keep the
            // separator, or the last phoneme of one line fuses with the first of the next
            ipa = String.join(" ", reader.lines().toList());
        }
        try {
            if (!espeak.waitFor(TIMEOUT_SECONDS, TimeUnit.SECONDS)) {
                espeak.destroyForcibly();
                throw new IOException(binary + " timed out on: " + words);
            }
        } catch (InterruptedException e) {
            espeak.destroyForcibly();
            Thread.currentThread().interrupt();
            throw new IOException(binary + " interrupted", e);
        }
        return ipa.replace("_", "").replaceAll("\\s+", " ").trim();
    }
}
