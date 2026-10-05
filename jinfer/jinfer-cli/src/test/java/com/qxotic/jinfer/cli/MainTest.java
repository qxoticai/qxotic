package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.hub.ModelStore;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
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
        assertEquals(0, Main.run(new String[] {verb, "-h"}, alternate.io, store));
        assertEquals(direct.out(), alternate.out());
        assertTrue(direct.out().contains("Usage:"));
        assertEquals("", direct.err());
        assertFalse(Files.exists(store.root()));
        assertFalse(direct.inputClosed);
        if (verb.equals("speak")) assertFalse(direct.out().contains("--port"));
    }

    /**
     * Each help screen is assembled from several sections; every option's description still starts
     * at one column, on the option's line or, for a long option, alone on the next.
     */
    @ParameterizedTest
    @ValueSource(
            strings = {
                "chat",
                "instruct",
                "server",
                "speak",
                "transcribe",
                "pull",
                "list",
                "cache-info"
            })
    void helpDescriptionsShareOneColumn(String verb) {
        var help = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {verb, "--help"}, help.io, ModelStore.of(dir)));
        String pad = " ".repeat(Options.HELP_COLUMN);
        List<String> lines = help.out().lines().toList();
        for (int i = 0; i < lines.size(); i++) {
            String line = lines.get(i);
            assertTrue(line.length() <= Options.HELP_WIDTH, verb + " overflows: " + line);
            if (!line.startsWith("  -")) continue;
            boolean inline =
                    line.length() > Options.HELP_COLUMN
                            && line.substring(0, Options.HELP_COLUMN).endsWith("  ")
                            && line.charAt(Options.HELP_COLUMN) != ' ';
            boolean below =
                    i + 1 < lines.size()
                            && lines.get(i + 1).startsWith(pad)
                            && lines.get(i + 1).charAt(Options.HELP_COLUMN) != ' ';
            assertTrue(inline || below, verb + " misaligned: " + line);
        }
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
        assertTrue(capture.err().contains("unknown command: chta"));
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
    void everyMissingInputFileReadsTheSame() {
        String missing = dir.resolve("absent").toString();
        for (String[] args :
                new String[][] {
                    {"transcribe", "-m", "unused.gguf", missing + ".wav"},
                    {"cache-info", missing + ".jkv"},
                    {"instruct", "-m", missing + ".gguf", "hi"}
                }) {
            var capture = new CliFixtures.Capture("");
            assertEquals(1, Main.run(args, capture.io, ModelStore.of(dir)));
            String file = args[args.length - (args[0].equals("instruct") ? 2 : 1)];
            assertEquals(
                    "jinfer " + args[0] + ": no such file: '" + file + "'", capture.err().strip());
        }
    }

    @Test
    void unknownOptionsPointToScopedHelpButBadValuesShowTheValue() {
        var unknown = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {"speak", "--wat"}, unknown.io, ModelStore.of(dir)));
        assertTrue(unknown.err().contains("speak --help' for available options."));
        var root = new CliFixtures.Capture("");
        assertEquals(2, Main.run(new String[] {"--wat"}, root.io, ModelStore.of(dir)));
        assertTrue(root.err().contains("--help' for available commands and options."));
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

    /**
     * Every option a command's help names is one that command accepts, and a command's help names
     * nothing it refuses: the help and the parser cannot drift apart.
     */
    @ParameterizedTest
    @ValueSource(strings = {"chat", "instruct", "server", "speak", "transcribe"})
    void helpNamesOnlyOptionsTheCommandAccepts(String verb) {
        var help = new CliFixtures.Capture("");
        assertEquals(0, Main.run(new String[] {verb, "--help"}, help.io, ModelStore.of(dir)));
        String operand =
                switch (verb) {
                    case "instruct", "speak" -> "hi";
                    case "transcribe" -> "-";
                    default -> null;
                };
        var flags =
                help.out()
                        .lines()
                        .filter(line -> line.startsWith("  -"))
                        .flatMap(
                                line ->
                                        java.util.regex.Pattern.compile("--[a-z-]+")
                                                .matcher(line)
                                                .results())
                        .map(java.util.regex.MatchResult::group)
                        .distinct()
                        .toList();
        assertFalse(flags.isEmpty(), verb + " help lists options");
        for (String flag : flags) {
            var args = new java.util.ArrayList<>(List.of(verb, "-m", "m"));
            if (operand != null) args.add(operand);
            args.addAll(List.of(flag, "1"));
            try {
                Options.parse(args.toArray(String[]::new));
            } catch (Options.UsageException e) {
                // a value or a switch may be wrong here; the option itself must be in scope
                for (String scope : List.of("does not apply", "unknown option", "must follow"))
                    assertFalse(
                            e.getMessage().contains(scope),
                            verb + " " + flag + ": " + e.getMessage());
            }
        }
        if (verb.equals("server"))
            assertFalse(help.out().contains("inline"), "server cannot route thoughts inline");
        if (verb.equals("speak") || verb.equals("server"))
            assertFalse(help.out().contains("--color"), verb + " never reads --color");
    }
}
