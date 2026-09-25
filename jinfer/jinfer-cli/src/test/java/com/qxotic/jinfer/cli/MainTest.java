package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class MainTest {
    @TempDir Path dir;

    @ParameterizedTest
    @ValueSource(
            strings = {
                "chat",
                "instruct",
                "prompt",
                "server",
                "serve",
                "speak",
                "transcribe",
                "pull",
                "list",
                "cache-info"
            })
    void everyVerbHasScopedHelpWithoutModelsOrCacheWrites(String verb) {
        var store = ModelStore.of(dir.resolve("absent-cache"));
        var direct = new CliFixtures.Capture("");
        var alternate = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {verb, "--help"}, direct.io, store));
        assertEquals(0, Main.run(new String[] {"help", verb}, alternate.io, store));
        assertEquals(direct.out(), alternate.out());
        assertTrue(direct.out().contains("Usage:"));
        assertEquals("", direct.err());
        assertFalse(Files.exists(store.root()));
        assertFalse(direct.inputClosed);
        if (verb.equals("speak")) assertFalse(direct.out().contains("--port"));
    }

    @Test
    void rootHelpAndVersionAreSuccessful() {
        for (String[] argv : new String[][] {{}, {"--help"}, {"--version"}}) {
            var capture = new CliFixtures.Capture("");
            assertEquals(0, Main.run(argv, capture.io, ModelStore.of(dir)));
            assertTrue(capture.out().contains("jinfer"));
            assertEquals("", capture.err());
        }
    }

    @Test
    void helpExplainsAMalformedInvocationWithoutResolvingAModel() {
        for (String[] args :
                new String[][] {
                    {"speak", "--speed", "fast", "--help"},
                    {"--temp", "oops", "chat", "--help"},
                    {"server", "--with", "invalid", "--help"},
                    {"instruct", "--unknown", "--help"}
                }) {
            var capture = new CliFixtures.Capture("");
            var store = ModelStore.of(dir.resolve("no-cache"));
            assertEquals(0, Main.run(args, capture.io, store), capture.err());
            assertTrue(capture.out().contains("Usage:"));
            assertEquals("", capture.err());
            assertFalse(Files.exists(store.root()));
        }
    }

    @Test
    void invalidArgumentsFailBeforeResolutionAndNameTheirCommand() {
        var capture = new CliFixtures.Capture("");
        Path cache = dir.resolve("cache");
        assertEquals(
                2,
                Main.run(
                        new String[] {"speak", "-m", "not-cached/repo:Q8_0", "Hi", "--speed", "0"},
                        capture.io,
                        ModelStore.of(cache)));
        assertTrue(capture.err().contains("jinfer speak:"));
        assertTrue(capture.err().contains("got 0"));
        assertFalse(
                capture.err().contains("--help"),
                "the range error already explains the correction");
        assertEquals("", capture.out());
        assertFalse(Files.exists(cache));
    }

    @Test
    void missingFileIsAnOperationalFailureAndDoesNotCloseStreams() {
        var capture = new CliFixtures.Capture("");
        assertEquals(
                1,
                Main.run(
                        new String[] {"cache-info", dir.resolve("missing.jkv").toString()},
                        capture.io,
                        ModelStore.of(dir)));
        assertTrue(capture.err().contains("no such file"));
        assertFalse(capture.err().contains("Exception in thread"));
        capture.io.out().println("still open");
        assertEquals("still open\n", capture.out().replace("\r\n", "\n"));
    }

    @Test
    void failedOutputNeverReportsSuccessEvenForHelp() {
        var capture = new CliFixtures.Capture("");
        var broken =
                new java.io.PrintStream(
                        new java.io.OutputStream() {
                            public void write(int value) throws java.io.IOException {
                                throw new java.io.IOException("closed");
                            }
                        });
        var io = new Main.IO(capture.io.in(), broken, capture.io.err());
        assertEquals(1, Main.run(new String[] {"--help"}, io, ModelStore.of(dir)));
        assertTrue(capture.err().contains("cannot write to stdout"));
        assertFalse(capture.err().contains("null"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"chat", "instruct", "server", "speak", "transcribe"})
    void missingModelsAreUsageErrorsForEveryApplication(String verb) {
        var capture = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {verb}, capture.io, ModelStore.of(dir)));
        assertTrue(capture.err().contains("--model"));
        assertTrue(capture.err().contains("specify --model"));
        assertEquals("", capture.out());
    }

    @Test
    void emptyPipedTextFailsBeforeTryingToLoadTheModel() {
        for (String verb : java.util.List.of("speak", "instruct")) {
            var capture = new CliFixtures.Capture(" \n\t");
            assertEquals(
                    2,
                    Main.run(
                            new String[] {verb, "-m", "missing-model", "-"},
                            capture.io,
                            ModelStore.of(dir)));
            assertTrue(capture.err().contains("non-blank text"));
            assertEquals("", capture.out());
            assertFalse(capture.inputClosed);
        }
    }

    @Test
    void unknownCommandsGetAConciseErrorInsteadOfARuntimeTrace() {
        var capture = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {"chta"}, capture.io, ModelStore.of(dir)));
        assertTrue(capture.err().contains("Unknown command: chta"));
        assertTrue(capture.err().lines().count() <= 3);
        assertEquals("", capture.out());
    }

    @Test
    void localModelErrorsAreShortAndDoNotTeachHubSyntax() {
        for (String verb : java.util.List.of("chat", "pull")) {
            var capture = new CliFixtures.Capture("");
            String missing = dir.resolve("missing model.gguf").toString();
            String[] args =
                    verb.equals("chat")
                            ? new String[] {verb, "-m", missing}
                            : new String[] {verb, missing};
            assertEquals(1, Main.run(args, capture.io, ModelStore.of(dir)));
            assertTrue(capture.err().contains("no such file: '" + missing + "'"));
            assertEquals(1, capture.err().lines().count());
            assertFalse(capture.err().contains("owner/repo"));
            assertEquals("", capture.out());
        }
    }

    @Test
    void unknownOptionsPointToScopedHelpButBadValuesShowTheValue() {
        var unknown = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {"speak", "--wat"}, unknown.io, ModelStore.of(dir)));
        assertTrue(unknown.err().contains("speak --help"));
        var prefix = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {"--wat", "speak"}, prefix.io, ModelStore.of(dir)));
        assertTrue(prefix.err().contains("speak --help"));
        var badValue = new CliFixtures.Capture("");
        assertEquals(
                2,
                Main.run(
                        new String[] {"instruct", "-m", "unused", "hello", "--top-p", "1.5"},
                        badValue.io,
                        ModelStore.of(dir)));
        assertEquals(
                "jinfer instruct: --top-p must be greater than 0 and at most 1; got 1.5\n",
                badValue.err().replace("\r\n", "\n"));
    }

    @Test
    void contextualIoErrorsKeepTheCauseAndIndentBackendDiagnostics() {
        var cause = new java.io.IOException("decoder exited 7:\ninvalid audio");
        var error = Main.failure("cannot decode audio 'clip.wav'", cause);
        assertEquals(
                "cannot decode audio 'clip.wav'\n  decoder exited 7:\n  invalid audio",
                error.getMessage());
        assertSame(cause, error.getCause());
    }

    @Test
    void failedStdinReadsNameTheInputBeingRead() {
        for (String verb : java.util.List.of("instruct", "transcribe")) {
            var capture = new CliFixtures.Capture("");
            var input =
                    new java.io.InputStream() {
                        public int read() throws java.io.IOException {
                            throw new java.io.IOException("input device closed");
                        }
                    };
            var io = new Main.IO(input, capture.io.out(), capture.io.err());
            assertEquals(
                    1,
                    Main.run(
                            new String[] {verb, "-m", "missing.gguf", "-"},
                            io,
                            ModelStore.of(dir)));
            String kind = verb.equals("instruct") ? "text" : "audio";
            assertTrue(capture.err().contains("cannot read " + kind + " from stdin"));
            assertTrue(capture.err().contains("\n  input device closed"));
            assertFalse(capture.err().contains("\tat "));
        }
    }
}
