package com.qxotic.jinfer.models.kokoro;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

final class KokoroPhonemizerTest {

    @Test
    void stripsEspeakArtifactsButPreservesInputPunctuation(@TempDir Path dir) throws IOException {
        KokoroPhonemizer phonemizer =
                stub(
                        dir,
                        "phonemes.sh",
                        "input=$(cat)\n"
                                + "case \"$input\" in\n"
                                + "  'Hello world') printf 'h_ə_(en)l_oʊ   wɝld\\n' ;;\n"
                                + "  again) printf '(en)əɡɛn\\n' ;;\n"
                                + "esac\n",
                        5);

        assertEquals(
                "h ə l oʊ wɝld , ( əɡɛn ) !", phonemizer.phonemize("  Hello   world, (again)!  "));
    }

    @Test
    void inputUnderscoresSurviveAlthoughEspeakUnderscoresDoNot(@TempDir Path dir)
            throws IOException {
        KokoroPhonemizer phonemizer =
                stub(
                        dir,
                        "underscore.sh",
                        "input=$(cat)\n"
                                + "if [ \"$input\" = a ]; then printf 'e_ɪ\\n"
                                + "'; else printf 'b_iː\\n"
                                + "'; fi\n",
                        5);

        assertEquals("e ɪ _ b iː", phonemizer.phonemize("a_b"));
    }

    @Test
    void contractionsAndHyphenatedWordsKeepTheirContext(@TempDir Path dir) throws IOException {
        KokoroPhonemizer phonemizer =
                stub(
                        dir,
                        "words.sh",
                        "input=$(cat)\n"
                                + "if [ \"$input\" = \"don't state-of-the-art\" ]; then"
                                + " printf 'context-kept\\n'; fi\n",
                        5);

        assertEquals("context-kept", phonemizer.phonemize("don't state-of-the-art"));
    }

    @Test
    void decimalPunctuationKeepsItsContext(@TempDir Path dir) throws IOException {
        KokoroPhonemizer phonemizer =
                stub(
                        dir,
                        "decimal.sh",
                        "input=$(cat)\n"
                            + "if [ \"$input\" = \"Value 3.14\" ]; then printf 'value three point"
                            + " one four\\n"
                            + "'; fi\n",
                        5);

        assertEquals("value three point one four !", phonemizer.phonemize("Value 3.14!"));
    }

    @Test
    void nonzeroExitIsReported(@TempDir Path dir) throws IOException {
        KokoroPhonemizer phonemizer = stub(dir, "broken.sh", "cat >/dev/null\nexit 3\n", 5);

        IOException error =
                assertThrows(IOException.class, () -> phonemizer.phonemize("broken input"));

        assertTrue(error.getMessage().contains("exited 3"), error.getMessage());
    }

    @Test
    void timeoutStopsAHungProcess(@TempDir Path dir) throws IOException {
        KokoroPhonemizer phonemizer = stub(dir, "hung.sh", "cat >/dev/null\nexec sleep 30\n", 1);

        IOException error = assertThrows(IOException.class, () -> phonemizer.phonemize("hello"));

        assertTrue(error.getMessage().contains("timed out"), error.getMessage());
    }

    @Test
    void constructionRejectsAnUnusableExecutable(@TempDir Path dir) throws IOException {
        Path script = script(dir, "bad-probe.sh", "exit 4\n");

        IOException error =
                assertThrows(IOException.class, () -> new KokoroPhonemizer(script.toString()));

        assertTrue(error.getMessage().contains("probe failed"), error.getMessage());
    }

    private static KokoroPhonemizer stub(Path dir, String name, String body, long timeoutSeconds)
            throws IOException {
        return new KokoroPhonemizer(
                script(dir, name, "if [ \"$1\" = --version ]; then exit 0; fi\n" + body).toString(),
                timeoutSeconds);
    }

    private static Path script(Path dir, String name, String body) throws IOException {
        Assumptions.assumeTrue(Files.isExecutable(Path.of("/bin/sh")), "needs /bin/sh");
        Path script = dir.resolve(name);
        Files.writeString(script, "#!/bin/sh\n" + body);
        assertTrue(script.toFile().setExecutable(true), "could not make test script executable");
        return script;
    }
}
