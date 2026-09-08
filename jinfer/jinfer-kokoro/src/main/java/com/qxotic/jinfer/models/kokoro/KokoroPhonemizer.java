package com.qxotic.jinfer.models.kokoro;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.util.Objects;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/** Minimal US-English eSpeak frontend for Kokoro's IPA vocabulary. */
public final class KokoroPhonemizer {

    private static final long DEFAULT_TIMEOUT_SECONDS = 5;
    private static final long PROBE_TIMEOUT_SECONDS = 2;

    private final String binary;
    private final long timeoutSeconds;

    KokoroPhonemizer(String binary) throws IOException {
        this(binary, DEFAULT_TIMEOUT_SECONDS);
    }

    KokoroPhonemizer(String binary, long timeoutSeconds) throws IOException {
        this.binary = Objects.requireNonNull(binary, "binary");
        if (timeoutSeconds <= 0) throw new IllegalArgumentException("timeout must be positive");
        this.timeoutSeconds = timeoutSeconds;
        probe();
    }

    /** Returns the first working eSpeak implementation on PATH, or {@code null} if none exists. */
    static KokoroPhonemizer tryCreate() {
        for (String name : new String[] {"espeak-ng", "espeak"}) {
            try {
                return new KokoroPhonemizer(name);
            } catch (IOException ignored) {
                if (Thread.currentThread().isInterrupted()) return null;
            }
        }
        return null;
    }

    /** Converts raw text to normalized IPA while preserving its punctuation. */
    public String phonemize(String text) throws IOException {
        Objects.requireNonNull(text, "text");
        var result = new StringBuilder();
        var run = new StringBuilder();
        for (int offset = 0; offset < text.length(); ) {
            int start = offset;
            int codePoint = text.codePointAt(offset);
            offset += Character.charCount(codePoint);
            boolean insideToken =
                    (codePoint == '\'' || codePoint == '’' || codePoint == '-')
                            && start > 0
                            && offset < text.length()
                            && Character.isLetter(text.codePointBefore(start))
                            && Character.isLetter(text.codePointAt(offset));
            insideToken |=
                    (codePoint == '.' || codePoint == ',')
                            && start > 0
                            && offset < text.length()
                            && Character.isDigit(text.codePointBefore(start))
                            && Character.isDigit(text.codePointAt(offset));
            if (isPunctuation(codePoint) && !insideToken) {
                flush(run, result);
                result.appendCodePoint(codePoint).append(' ');
            } else {
                run.appendCodePoint(codePoint);
            }
        }
        flush(run, result);
        return normalizeWhitespace(result.toString());
    }

    private void flush(StringBuilder run, StringBuilder result) throws IOException {
        String words = normalizeWhitespace(run.toString());
        run.setLength(0);
        if (!words.isEmpty()) result.append(ipaRun(words)).append(' ');
    }

    private String ipaRun(String words) throws IOException {
        Process espeak =
                new ProcessBuilder(binary, "--ipa", "-q", "-v", "en-us", "--stdin")
                        .redirectError(ProcessBuilder.Redirect.DISCARD)
                        .start();
        var output =
                new FutureTask<>(
                        () ->
                                new String(
                                        espeak.getInputStream().readAllBytes(),
                                        StandardCharsets.UTF_8));
        Thread.ofVirtual().start(output);
        try {
            try (var stdin = espeak.getOutputStream()) {
                stdin.write((words + "\n").getBytes(StandardCharsets.UTF_8));
            }
            if (!espeak.waitFor(timeoutSeconds, TimeUnit.SECONDS))
                throw new IOException(binary + " timed out while phonemizing: " + words);
            String ipa = output.get(timeoutSeconds, TimeUnit.SECONDS);
            if (espeak.exitValue() != 0)
                throw new IOException(
                        binary + " exited " + espeak.exitValue() + " while phonemizing: " + words);
            return normalizeWhitespace(
                    ipa.replace("_", " ").replaceAll("(?i)\\([a-z]{2,3}(?:-[a-z0-9]+)*\\)", " "));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(binary + " was interrupted", e);
        } catch (TimeoutException e) {
            throw new IOException(binary + " timed out while reading IPA output", e);
        } catch (ExecutionException e) {
            throw new IOException(binary + " IPA output could not be read", e.getCause());
        } finally {
            if (espeak.isAlive()) espeak.destroyForcibly();
        }
    }

    private void probe() throws IOException {
        Process process;
        try {
            process =
                    new ProcessBuilder(binary, "--version")
                            .redirectOutput(ProcessBuilder.Redirect.DISCARD)
                            .redirectError(ProcessBuilder.Redirect.DISCARD)
                            .start();
        } catch (IOException e) {
            throw new IOException("eSpeak executable is unavailable: " + binary, e);
        }
        try {
            if (!process.waitFor(PROBE_TIMEOUT_SECONDS, TimeUnit.SECONDS))
                throw new IOException("eSpeak executable probe timed out: " + binary);
            if (process.exitValue() != 0)
                throw new IOException(
                        "eSpeak executable probe failed (exit "
                                + process.exitValue()
                                + "): "
                                + binary);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException("eSpeak executable probe was interrupted: " + binary, e);
        } finally {
            if (process.isAlive()) process.destroyForcibly();
        }
    }

    private static boolean isPunctuation(int codePoint) {
        return switch (Character.getType(codePoint)) {
            case Character.CONNECTOR_PUNCTUATION,
                    Character.DASH_PUNCTUATION,
                    Character.START_PUNCTUATION,
                    Character.END_PUNCTUATION,
                    Character.INITIAL_QUOTE_PUNCTUATION,
                    Character.FINAL_QUOTE_PUNCTUATION,
                    Character.OTHER_PUNCTUATION ->
                    true;
            default -> false;
        };
    }

    private static String normalizeWhitespace(String text) {
        return text.replaceAll("(?U)\\s+", " ").trim();
    }
}
