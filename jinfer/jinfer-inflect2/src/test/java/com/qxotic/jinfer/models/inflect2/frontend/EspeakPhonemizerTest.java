package com.qxotic.jinfer.models.inflect2.frontend;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.models.inflect2.Symbols;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;

/**
 * Pins the punctuation-run split, not the phonemes: espeak's IPA is version-dependent, so the
 * durable contract is that every mark survives exactly once in source order and that every
 * punctuation-free span reaches espeak - including the trailing one, which the loop is one missing
 * flush away from dropping.
 *
 * <p>Skips when espeak-ng is absent; the lexicon front end is tested separately.
 */
class EspeakPhonemizerTest {

    private static Phonemizer espeak() {
        Phonemizer espeak = EspeakPhonemizer.tryCreate();
        Assumptions.assumeTrue(espeak != null, "espeak-ng is not installed");
        return espeak;
    }

    /** The symbols of {@code tokens} that are punctuation, in order - blanks and phonemes out. */
    private static List<Integer> marks(int[] tokens) {
        List<Integer> marks = new ArrayList<>();
        for (int token : tokens) {
            for (char mark : new char[] {',', '.', ';', '!', '?', ':'}) {
                if (token == Symbols.idOf(mark)) marks.add(token);
            }
        }
        return marks;
    }

    @Test
    void aRunStartingWithADashIsTextNotAnOption() throws IOException {
        // "-5 degrees" normalizes to "-five degrees"; as an argv element getopt read "-f"
        int[] tokens = espeak().phonemize("-five degrees");
        assertTrue(tokens.length > 4, "both words phonemized: " + tokens.length);
    }

    @Test
    void stderrIsNeverSpokenAndAFailedRunIsAnError() throws IOException {
        Assumptions.assumeTrue(Files.isExecutable(Path.of("/bin/sh")), "needs a shell");
        Path dir = Files.createTempDirectory("espeak-stub");
        Path chatty = dir.resolve("chatty.sh"), broken = dir.resolve("broken.sh");
        Files.writeString(
                chatty,
                "#!/bin/sh\n"
                        + "cat >/dev/null\n"
                        + "echo 'warning: no voice data' >&2\n"
                        + "echo 'h\u0259l\u02c8o\u028a'\n");
        Files.writeString(broken, "#!/bin/sh\ncat >/dev/null\necho 'error: option' >&2\nexit 2\n");
        chatty.toFile().setExecutable(true);
        broken.toFile().setExecutable(true);
        int[] tokens = new EspeakPhonemizer(chatty.toString()).phonemize("hello");
        // the IPA line alone is ~11 tokens; with the warning spoken too it would be over 50
        assertTrue(tokens.length > 0 && tokens.length < 20, "only the IPA line: " + tokens.length);
        IOException e =
                Assertions.assertThrows(
                        IOException.class,
                        () -> new EspeakPhonemizer(broken.toString()).phonemize("hello"));
        assertTrue(e.getMessage().contains("exited 2"), e.getMessage());
    }

    @Test
    void everyMarkSurvivesOnceInSourceOrder() throws IOException {
        int[] tokens = espeak().phonemize("Yes, of course; that is right. Is it?");
        assertEquals(
                List.of(Symbols.idOf(','), Symbols.idOf(';'), Symbols.idOf('.'), Symbols.idOf('?')),
                marks(tokens));
    }

    @Test
    void theTrailingRunIsFlushed() throws IOException {
        // no punctuation at all: the whole text is one open run, emitted only by the final flush
        int[] tokens = espeak().phonemize("one two three");
        assertTrue(tokens.length > 6, "a three-word run must phonemize to more than blanks");
        assertTrue(marks(tokens).isEmpty(), "nothing punctuates this");
    }

    @Test
    void aRunIsPhonemizedAsOneSpan() throws IOException {
        Phonemizer espeak = espeak();
        // espeak reads context: per word it stamps a primary stress on every function word, so a
        // batched run is NOT the concatenation of its words. If this ever matches, the split
        // regressed to one subprocess per word.
        int[] run = espeak.phonemize("the cat can see the dog");
        int[] words = concat(espeak, "the", "cat", "can", "see", "the", "dog");
        assertTrue(run.length != words.length || !Arrays.equals(run, words));
    }

    @Test
    void espeakReceivesATerminatedLine() throws IOException {
        // espeak reads stdin a line at a time. An unterminated final line is a word FRAGMENT to
        // it, and it answers with a fragment: "world" came back "wˈɜːl", "five" as "fˈɪv",
        // "hello" as "hˈɛl", "em" as "ˈiː" - 46 of 56 common words measured wrong, once per run.
        Assumptions.assumeTrue(Files.isExecutable(Path.of("/bin/sh")), "needs a shell");
        Path dir = Files.createTempDirectory("espeak-stub");
        Path seen = dir.resolve("seen.txt"), stub = dir.resolve("stub.sh");
        Files.writeString(stub, "#!/bin/sh\ncat > '" + seen + "'\necho 'həlˈoʊ'\n");
        stub.toFile().setExecutable(true);

        new EspeakPhonemizer(stub.toString()).phonemize("hello world");

        String sent = Files.readString(seen);
        assertTrue(sent.endsWith("\n"), "espeak was handed an unterminated line: " + sent);
        assertEquals("hello world", sent.strip(), "only the run itself, plus the terminator");
    }

    @Test
    void aWordIsTheSameWordWhereverItFallsInTheRun() throws IOException {
        // The version-independent form of the same defect: whatever espeak thinks "world"
        // sounds like, it must think so in final position too. Unterminated, the halves diverged.
        int[] tokens = espeak().phonemize("world world");
        List<Integer> spoken = new ArrayList<>();
        for (int token : tokens) if (token != 0) spoken.add(token); // drop the blanks
        int split = spoken.indexOf(Symbols.idOf(' '));
        assertTrue(split > 0, "the run must keep a separator between the two words");
        assertEquals(
                spoken.subList(0, split),
                spoken.subList(split + 1, spoken.size()),
                "the same word twice must phonemize the same twice");
    }

    private static int[] concat(Phonemizer espeak, String... words) throws IOException {
        int[] tokens = new int[0];
        for (String word : words) {
            int[] one = espeak.phonemize(word);
            int[] grown = Arrays.copyOf(tokens, tokens.length + one.length);
            System.arraycopy(one, 0, grown, tokens.length, one.length);
            tokens = grown;
        }
        return tokens;
    }
}
