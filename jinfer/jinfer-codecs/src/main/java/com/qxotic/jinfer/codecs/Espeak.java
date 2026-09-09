package com.qxotic.jinfer.codecs;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;

/**
 * espeak-ng as a grapheme-to-phoneme tool: one punctuation-free run of words in, its IPA out. The
 * one place jinfer talks to espeak; the speech ports build their {@link
 * com.qxotic.jinfer.Phonemizer} over it.
 *
 * <p>Optional, like the ffmpeg codecs beside it: {@link #find()} probes {@code PATH} once, and a
 * port that can do without (a pronunciation lexicon) asks rather than requires.
 */
public final class Espeak {

    private static final Duration PROBE_TIMEOUT = Duration.ofSeconds(2);
    private static final Duration TIMEOUT = Duration.ofSeconds(5);
    private static final int MAX_OUTPUT_BYTES = 1 << 20;

    /** The character {@link #tiedIpa} joins a multi-letter phoneme with; never in espeak's IPA. */
    public static final String TIE = "^";

    /** espeak's language-switch markers, {@code (en)} {@code (fr-fr)}: annotation, not IPA. */
    private static final Pattern LANGUAGE_MARKER =
            Pattern.compile("(?i)\\([a-z]{2,3}(?:-[a-z0-9]+)*\\)");

    private final String binary;
    private final Duration timeout;

    Espeak(String binary, Duration timeout) { // package-private: tests point it at a stand-in
        this.binary = Objects.requireNonNull(binary, "binary");
        if (timeout.isNegative() || timeout.isZero())
            throw new IllegalArgumentException("timeout must be positive: " + timeout);
        this.timeout = timeout;
    }

    /** The first working {@code espeak-ng} or {@code espeak} on PATH, probed once. */
    public static Optional<Espeak> find() {
        for (String name : new String[] {"espeak-ng", "espeak"}) {
            try {
                Subprocess.run(List.of(name, "--version"), null, PROBE_TIMEOUT, 64 << 10);
                return Optional.of(new Espeak(name, TIMEOUT));
            } catch (IOException notThisOne) {
                // not installed under this name, or not answering: try the next
            }
        }
        return Optional.empty();
    }

    /**
     * IPA for one punctuation-free run of words, in an espeak voice ({@code en-us}, {@code fr-fr},
     * {@code cmn}). Words keep the writer's case - espeak reads capitals as information ("GraalVM"
     * is "graal vee em", "graalvm" one mangled word).
     *
     * <p>Unchecked: the tool was probed at load, so a failure here is a broken installation or a
     * hung process, not a condition a caller handles.
     */
    public String ipa(String run, String language) {
        return tiedIpa(run, language).replace(TIE, "");
    }

    /**
     * As {@link #ipa} with espeak's multi-letter phonemes joined by {@link #TIE}: an affricate is
     * {@code d^ʒ} and the diphthong in "day" {@code e^ɪ}, while the {@code tʃ} of "nightshirt"
     * stays two phonemes. For a caller that rewrites phonemes and must not merge across a boundary
     * espeak did not draw. Stripping the tie gives {@link #ipa}'s form exactly.
     */
    public String tiedIpa(String run, String language) {
        Objects.requireNonNull(run, "run");
        Objects.requireNonNull(language, "language");
        // The run goes in on stdin, never as an argument: one starting with '-' ("-five
        // degrees") would be parsed as an option. The trailing newline is NOT cosmetic: espeak
        // reads stdin a line at a time and phonemizes the last word of an unterminated line as a
        // word FRAGMENT - "world" came back "wˈɜːl", "five" as "fˈɪv", "hello" as "hˈɛl".
        List<String> command =
                List.of(binary, "--ipa", "-q", "-v", language, "--tie=" + TIE, "--stdin");
        byte[] stdin = (run + "\n").getBytes(StandardCharsets.UTF_8);
        byte[] out;
        try {
            out = Subprocess.run(command, stdin, timeout, MAX_OUTPUT_BYTES);
        } catch (IOException e) {
            throw new UncheckedIOException(binary + " failed on: " + run, e);
        }
        // A run can span several output lines: keep a separator, or adjacent phonemes would fuse
        // at espeak's line boundary. '_' is espeak's own phoneme separator, never a phoneme.
        String ipa = new String(out, StandardCharsets.UTF_8).replace('\n', ' ').replace("_", "");
        return LANGUAGE_MARKER.matcher(ipa).replaceAll("").replaceAll("\\s+", " ").trim();
    }
}
