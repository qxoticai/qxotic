package com.qxotic.jinfer.models.parakeet;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Transcription;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.text.Normalizer;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * A reference transcript ({@code golden.tsv}): what is said, from a source independent of every
 * engine, and each whitespace-separated word's start time. Checks are deliberately tolerant (a few
 * word errors, a couple of frames of timing), so quantization and float drift pass while a real
 * regression does not.
 */
record Golden(String clip, String text, double[] starts, double timingTolerance) {

    /** Word errors allowed: one, or 5% of the words, whichever is more. */
    private static int allowedErrors(int words) {
        return Math.max(1, words / 20);
    }

    /** Two encoder frames: on a clip, every engine and quant measured agrees within one. */
    static final double CLIP_TIMING = 0.16;

    /**
     * Four encoder frames: inside longer audio, a clip's words start up to three and a bit frames
     * off the clip's own times, its first word most of all, in parakeet.cpp as here.
     */
    static final double CONTEXT_TIMING = 0.32;

    private static final Map<String, Golden> ALL = load();

    static Golden of(String clip) {
        Golden golden = ALL.get(clip);
        if (golden == null) throw new IllegalArgumentException("no golden for " + clip);
        return golden;
    }

    /** Golden data is float seconds, as parakeet.cpp reports it. */
    private static double seconds(Duration offset) {
        return offset.toNanos() / 1e9;
    }

    static List<String> clips() {
        return List.copyOf(ALL.keySet());
    }

    /**
     * The goldens back to back, each starting {@code offsets[i]} seconds into the audio; timed to
     * {@link #CONTEXT_TIMING}.
     */
    static Golden concat(List<Golden> goldens, double[] offsets) {
        StringBuilder text = new StringBuilder();
        List<Double> starts = new ArrayList<>();
        for (int i = 0; i < goldens.size(); i++) {
            if (i > 0) text.append(' ');
            text.append(goldens.get(i).text());
            for (double start : goldens.get(i).starts()) starts.add(start + offsets[i]);
        }
        return new Golden(
                "concat",
                text.toString(),
                starts.stream().mapToDouble(d -> d).toArray(),
                CONTEXT_TIMING);
    }

    /**
     * {@code transcription} of {@code audioSeconds} of audio says this golden: well-formed tokens,
     * at most {@link #allowedErrors} word errors, and the words it shares with the golden start
     * within {@link #timingTolerance} of the golden times shifted by {@code offset}.
     */
    void assertHeardIn(Transcription transcription, double audioSeconds, double offset) {
        double previous = 0;
        for (Transcription.Token token : transcription.tokens()) {
            assertTrue(seconds(token.start()) >= previous, clip + ": token starts are monotone");
            assertTrue(
                    seconds(token.end()) <= audioSeconds + 1e-9,
                    clip + ": tokens end inside the audio");
            assertTrue(
                    token.confidence() >= 0 && token.confidence() <= 1,
                    clip + ": confidence in [0,1]");
            previous = seconds(token.start());
        }

        List<String> reference = words(text), heard = words(transcription.text());
        int errors = editDistance(reference, heard);
        assertTrue(
                errors <= allowedErrors(reference.size()),
                clip
                        + ": "
                        + errors
                        + " word errors (allowed "
                        + allowedErrors(reference.size())
                        + ")\n  golden: "
                        + text
                        + "\n  heard:  "
                        + transcription.text());

        // Timing, word by word: align the golden's words with the heard ones (tokens grouped at
        // their word-initial space) and compare the starts of the pairs that agree.
        String[] goldenWords = text.split(" ");
        assertEquals(goldenWords.length, starts.length, clip + ": golden has a start per word");
        List<String> heardKeys = new ArrayList<>();
        List<Double> heardStarts = new ArrayList<>();
        for (Transcription.Token token : transcription.tokens()) {
            if (heardKeys.isEmpty() || token.text().startsWith(" ")) {
                heardKeys.add(key(token.text()));
                heardStarts.add(seconds(token.start()));
            } else {
                int last = heardKeys.size() - 1;
                heardKeys.set(last, heardKeys.get(last) + key(token.text()));
            }
        }
        String[] goldenKeys = Arrays.stream(goldenWords).map(Golden::key).toArray(String[]::new);
        int[][] pairs = matches(goldenKeys, heardKeys.toArray(String[]::new));
        assertTrue(
                pairs.length >= 0.8 * goldenWords.length,
                clip + ": only " + pairs.length + " of " + goldenWords.length + " words align");
        for (int[] pair : pairs) {
            double expected = starts[pair[0]] + offset, actual = heardStarts.get(pair[1]);
            assertEquals(
                    expected,
                    actual,
                    timingTolerance + 1e-9,
                    clip + ": '" + goldenWords[pair[0]] + "' starts at " + actual + "s");
        }
    }

    /** Lowercased NFC words of letters, digits and apostrophes, in any script. */
    private static List<String> words(String text) {
        String cleaned =
                Normalizer.normalize(text.toLowerCase(Locale.ROOT), Normalizer.Form.NFC)
                        .replaceAll("[^\\p{L}\\p{N}']+", " ")
                        .strip();
        return cleaned.isEmpty() ? List.of() : List.of(cleaned.split(" "));
    }

    private static String key(String word) {
        return String.join("", words(word));
    }

    private static int editDistance(List<String> reference, List<String> heard) {
        int[] previous = new int[heard.size() + 1], current = new int[heard.size() + 1];
        for (int j = 0; j <= heard.size(); j++) previous[j] = j;
        for (int i = 1; i <= reference.size(); i++) {
            current[0] = i;
            for (int j = 1; j <= heard.size(); j++) {
                int substitute =
                        previous[j - 1] + (reference.get(i - 1).equals(heard.get(j - 1)) ? 0 : 1);
                current[j] = Math.min(substitute, Math.min(previous[j] + 1, current[j - 1] + 1));
            }
            int[] swap = previous;
            previous = current;
            current = swap;
        }
        return previous[heard.size()];
    }

    /** Index pairs of equal words on a longest-common-subsequence alignment. */
    private static int[][] matches(String[] a, String[] b) {
        int[][] lcs = new int[a.length + 1][b.length + 1];
        for (int i = a.length - 1; i >= 0; i--)
            for (int j = b.length - 1; j >= 0; j--)
                lcs[i][j] =
                        a[i].equals(b[j])
                                ? lcs[i + 1][j + 1] + 1
                                : Math.max(lcs[i + 1][j], lcs[i][j + 1]);
        List<int[]> pairs = new ArrayList<>();
        for (int i = 0, j = 0; i < a.length && j < b.length; ) {
            if (a[i].equals(b[j])) pairs.add(new int[] {i++, j++});
            else if (lcs[i + 1][j] >= lcs[i][j + 1]) i++;
            else j++;
        }
        return pairs.toArray(int[][]::new);
    }

    private static Map<String, Golden> load() {
        Map<String, Golden> goldens = new LinkedHashMap<>();
        try (var in = Golden.class.getResourceAsStream("golden.tsv");
                var reader =
                        new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8))) {
            for (String line; (line = reader.readLine()) != null; ) {
                if (line.isBlank() || line.startsWith("#")) continue;
                String[] fields = line.split("\t");
                double[] starts =
                        Arrays.stream(fields[2].split(" "))
                                .mapToDouble(Double::parseDouble)
                                .toArray();
                goldens.put(fields[0], new Golden(fields[0], fields[1], starts, CLIP_TIMING));
            }
        } catch (IOException e) {
            throw new UncheckedIOException(e);
        }
        return goldens;
    }
}
