package com.qxotic.jinfer.codecs;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Arrays;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Pins the driver, not the phonemes: what espeak is handed, what of its output is IPA, and how it
 * fails. A shell script stands in for espeak in every test but the last two, which need the real
 * thing and skip without it.
 */
final class EspeakTest {

    @Test
    void stderrIsNeverIpaAndAFailedRunIsAnError(@TempDir Path dir) throws IOException {
        Espeak chatty =
                stub(dir, "chatty.sh", "echo 'warning: no voice data' >&2\necho 'həlˈoʊ'\n");
        Espeak broken = stub(dir, "broken.sh", "echo 'error: option' >&2\nexit 2\n");

        assertEquals("həlˈoʊ", chatty.ipa("hello", "en-us"));
        UncheckedIOException e =
                assertThrows(UncheckedIOException.class, () -> broken.ipa("hello", "en-us"));
        assertTrue(e.getCause().getMessage().contains("exited 2"), e.getCause().getMessage());
    }

    @Test
    void aHungProcessIsStoppedByTheTimeout(@TempDir Path dir) throws IOException {
        Espeak hung = stub(dir, "hung.sh", "exec sleep 30\n", Duration.ofMillis(200));

        UncheckedIOException e =
                assertThrows(UncheckedIOException.class, () -> hung.ipa("hello", "en-us"));

        assertTrue(e.getCause().getMessage().contains("timed out"), e.getCause().getMessage());
    }

    @Test
    void theRunGoesInOnStdinAsOneTerminatedLine(@TempDir Path dir) throws IOException {
        Path seen = dir.resolve("seen.txt");
        Espeak espeak = stub(dir, "seen.sh", "cat > '" + seen + "'\necho 'x'\n");

        // starts with '-': as an argv element getopt would have read "-f"
        espeak.ipa("-five degrees", "en-us");

        String sent = Files.readString(seen);
        assertTrue(sent.endsWith("\n"), "espeak was handed an unterminated line: " + sent);
        assertEquals("-five degrees", sent.strip(), "only the run itself, plus the terminator");
    }

    @Test
    void theLanguageIsTheEspeakVoice(@TempDir Path dir) throws IOException {
        Espeak espeak =
                stub(dir, "voice.sh", "if [ \"$4\" = es ]; then echo 'ola'; else exit 2; fi\n");
        assertEquals("ola", espeak.ipa("hola", "es"));
    }

    @Test
    void artifactsAreStrippedAndLinesAreJoined(@TempDir Path dir) throws IOException {
        Espeak espeak = stub(dir, "artifacts.sh", "printf 'h_ə_(en)l_oʊ  \\nwɝld (fr-fr)\\n'\n");
        assertEquals("həloʊ wɝld", espeak.ipa("hello world", "en-us"));
    }

    @Test
    void aWordIsTheSameWordWhereverItFallsInTheRun() {
        // The version-independent form of the fragment bug: whatever espeak thinks "world" sounds
        // like, it must think so in final position too. Unterminated, the halves diverged.
        Espeak espeak = installed();
        String[] halves = espeak.ipa("world world", "en-us").split(" ");
        assertEquals(2, halves.length, Arrays.toString(halves));
        assertEquals(halves[0], halves[1]);
    }

    @Test
    void tiedIpaMarksMultiLetterPhonemesAndStripsToThePlainForm() {
        Espeak espeak = installed();
        String tied = espeak.ipa("headjoint nightshirt", "en-us", "^");
        assertTrue(tied.contains("d^ʒ"), "the affricate is tied: " + tied);
        assertTrue(tied.contains("tʃ"), "t then ʃ across a boundary is not: " + tied);
        assertEquals(espeak.ipa("headjoint nightshirt", "en-us"), tied.replace("^", ""));
    }

    @Test
    void theWritersCaseReachesEspeak() {
        Espeak espeak = installed();
        // espeak reads capitals as information; the two must not come back the same
        assertFalse(espeak.ipa("GraalVM", "en-us").equals(espeak.ipa("graalvm", "en-us")));
    }

    private static Espeak installed() {
        return Espeak.find().orElseGet(() -> Assumptions.abort("espeak-ng is not installed"));
    }

    private static Espeak stub(Path dir, String name, String body) throws IOException {
        return stub(dir, name, body, Duration.ofSeconds(5));
    }

    /** A shell script standing in for espeak; skips the test when there is no shell. */
    private static Espeak stub(Path dir, String name, String body, Duration timeout)
            throws IOException {
        Assumptions.assumeTrue(Files.isExecutable(Path.of("/bin/sh")), "needs a shell");
        Path script = dir.resolve(name);
        Files.writeString(script, "#!/bin/sh\n" + body);
        assertTrue(script.toFile().setExecutable(true), "could not make the script executable");
        return new Espeak(script.toString(), timeout);
    }
}
