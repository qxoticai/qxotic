package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.*;

import com.qxotic.jinfer.RuntimeFlags;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampling;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class OptionsTest {
    @Test
    void modelSettingsHaveIdenticalMeaningOnEitherSideOfTheVerb() {
        for (String[] argv :
                new String[][] {
                    {"-m", "not-downloaded.gguf", "--temp", "0.3", "chat"},
                    {"chat", "--model=not-downloaded.gguf", "--temp=0.3"},
                    {"-m", "not-downloaded.gguf", "chat", "--temp", "0.3"}
                }) {
            Options o = Options.parse(argv);
            assertEquals("chat", o.command);
            assertEquals("not-downloaded.gguf", o.modelRef);
            assertEquals(0.3f, o.temperature);
            assertFalse(o.legacy);
        }
        Options last =
                Options.parse(
                        "-m", "first", "--temp", "0.8", "chat", "-m", "second", "--temp", "0");
        assertEquals("second", last.modelRef);
        assertEquals(0f, last.temperature);
    }

    @Test
    void defaultsRemainUnsetUntilTheModelSuppliesThem() {
        Options o = Options.parse("instruct", "-m", "missing", "hi");
        assertNull(o.temperature);
        assertNull(o.contextCapacity);
        assertNull(o.batchCapacity);
        assertNull(o.threads);
        assertEquals(0.8f, o.sampling(LoadedModel.SamplingDefaults.NONE).temperature());
        assertTrue(o.stream);
        var recommended = new LoadedModel.SamplingDefaults(0.2f, 0.7f, 12, 0.1f);
        assertEquals(0.2f, o.sampling(recommended).temperature());
        assertEquals(12, o.sampling(recommended).topK());
        Options explicit = Options.parse("instruct", "-m", "missing", "hi", "--temp", "0");
        assertEquals(0f, explicit.sampling(recommended).temperature());
        assertEquals(12, explicit.sampling(recommended).topK());
    }

    @Test
    void valuesAreNeverMistakenForCommandsOrOptions() {
        assertEquals("chat", Options.parse("-m", "chat", "server").modelRef);
        assertEquals("--server", Options.parse("instruct", "-m", "m", "--", "--server").input);
        assertEquals("--chat", Options.parse("-m", "m", "--prompt", "--chat").input);
        assertEquals("false", Options.parse("speak", "-m", "m", "--stream", "false").input);
        assertEquals("--help", Options.parse("instruct", "-m", "m", "-p", "--help").input);
    }

    @Test
    void companionsSplitOnlyAtTheFirstEqualsAndDoNotResolveWhileParsing() {
        String path = "C:\\Models\\O'Brien こんにちは=a.gguf";
        Options o =
                Options.parse(
                        "--with",
                        "model=repo/model:Q8_0",
                        "speak",
                        "--with",
                        "voice=" + path,
                        "Hi");
        assertEquals("repo/model:Q8_0", o.modelRef);
        assertEquals(path, o.companionRefs.get("voice"));
        assertThrows(
                Options.Usage.class,
                () ->
                        Options.parse(
                                "speak", "-m", "m", "Hi", "--with", "voice=a", "--with",
                                "voice=b"));
        for (String bad : List.of("voice", "=file", "voice=", "voice=auto"))
            assertThrows(
                    Options.Usage.class,
                    () -> Options.parse("speak", "-m", "m", "Hi", "--with", bad));
    }

    @Test
    void mmprojUsesTheSameCompanionValidationAsWithMedia() {
        assertEquals(
                Options.parse("chat", "-m", "m", "--with", "media=projector.gguf").companionRefs,
                Options.parse("chat", "-m", "m", "--mmproj", "projector.gguf").companionRefs);
        for (String value : List.of("", " ", "auto")) {
            assertThrows(
                    Options.Usage.class, () -> Options.parse("chat", "-m", "m", "--mmproj", value));
        }
        assertThrows(
                Options.Usage.class,
                () -> Options.parse("chat", "-m", "m", "--with", "media=a", "--mmproj", "b"));
    }

    @Test
    void aliasesAndLegacySyntaxShareCommandsButPreserveLegacyAudioDefaults() {
        assertEquals("server", Options.parse("serve", "-m", "m").command);
        assertEquals("instruct", Options.parse("prompt", "-m", "m", "hi").command);
        assertEquals("chat", Options.parse("-m", "m", "--chat").command);
        assertEquals("server", Options.parse("-m", "m", "--server").command);
        assertFalse(Options.parse("-m", "m", "-p", "hi", "--stream", "false").stream);
        Options speech = Options.parse("speak", "-m", "m", "hi");
        assertTrue(speech.speech.play);
        assertNull(speech.speech.output);
        assertEquals(
                Path.of("output.wav"), Options.parse("-m", "m", "--speak", "hi").speech.output);
        assertTrue(Options.parse("-m", "m", "--transcribe", "-").transcription.rawPcm);
        assertFalse(Options.parse("transcribe", "-m", "m", "-").transcription.rawPcm);
        assertTrue(Options.parse("transcribe", "-m", "m", "-", "--raw-pcm").transcription.rawPcm);
        assertEquals(
                "nord",
                Options.parse("transcribe", "-m", "m", "-", "--raw-pcm", "--theme", "nord")
                        .transcription
                        .theme
                        .name());
    }

    @Test
    void argumentValidationDoesNotNeedFilesOrNetwork() {
        for (String[] argv :
                new String[][] {
                    {"speak", "-m", "uncached/repo:Q8_0", "Hi", "--speed", "0"},
                    {"instruct", "-m", "nonexistent"},
                    {"chat", "-m", "m", "--port", "9000"},
                    {"speak", "-m", "m", "Hi", "--temp", "0.3"},
                    {"server", "-m", "m", "--system-prompt", "Hi"},
                    {"list", "-m", "m"},
                    {"--port", "9000", "server", "-m", "m"},
                    {"chat", "-m", "m", "--server"},
                    {"instruct", "-m", "m", "hello", "world"},
                    {"speak", "-m", "m", "hi", "--play", "--stream"},
                    {"speak", "-m", "m", "hi", "--play", "--output", "-"},
                    {"speak", "-m", "m", "hi", "--stream", "--output", "out.wav"},
                    {"speak", "-m", "m", " "},
                    {"transcribe", "-m", "m", "file.wav", "--raw-pcm"},
                    {"chat", "-m", "m", "--batch-capacity", "0"},
                    {"chat", "-m", "m", "--threads", "0"},
                    {"chat", "-m", "m", "--context-capacity", "-1"},
                    {"chat", "-m", "m", "--top-p", "1.7"},
                    {"chat", "-m", "m", "--temp", "NaN"},
                    {"chat", "-m", "m", "--max-reasoning-tokens", "-2"},
                    {"chat", "-m", "m", "--cache", "c.jkv"},
                    {"instruct", "-m", "m", "hi", "--raw-prompt", "--cache", "c.jkv"},
                    {"instruct", "-m", "m", "hi", "--raw-prompt", "--system-prompt", "terse"}
                })
            assertThrows(Options.Usage.class, () -> Options.parse(argv), String.join(" ", argv));
    }

    @Test
    void rawPromptRestrictionsAreSharedByInstructAndServer() {
        for (String command : List.of("instruct", "server")) {
            var args = new java.util.ArrayList<>(List.of(command, "-m", "unused", "--raw-prompt"));
            if (command.equals("instruct")) args.add("hello");
            for (String[] conflict :
                    new String[][] {
                        {"--think", "off"}, {"--max-reasoning-tokens", "8"},
                        {"--reasoning-cutoff-message", "enough"}, {"--cache", "out.jkv"}
                    }) {
                var invalid = new java.util.ArrayList<>(args);
                invalid.addAll(List.of(conflict));
                assertThrows(
                        Options.Usage.class, () -> Options.parse(invalid.toArray(String[]::new)));
            }
            args.addAll(List.of("--cache-ro", "existing.jkv"));
            assertDoesNotThrow(() -> Options.parse(args.toArray(String[]::new)));
        }
    }

    @Test
    void helpInsideAValueOrLiteralInputDoesNotHideAnEarlierError() {
        for (String[] args :
                new String[][] {
                    {"chat", "--temp", "oops", "--model", "--help"},
                    {"instruct", "-m", "unused", "--temp", "oops", "--", "--help"}
                }) {
            var error = assertThrows(Options.Usage.class, () -> Options.parse(args));
            assertTrue(error.getMessage().contains("--temp"));
        }
    }

    @Test
    void unnamedOperationalFailuresStillHaveADiagnostic() {
        assertEquals("IOException", Options.rootMessage(new java.io.IOException()));
        assertEquals(
                "missing file",
                Options.rootMessage(
                        new java.io.UncheckedIOException(new java.io.IOException("missing file"))));
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "-1", "NaN", "Infinity", "fast"})
    void invalidSpeechSpeedNamesItsFlag(String speed) {
        var e =
                assertThrows(
                        Options.Usage.class,
                        () -> Options.parse("speak", "-m", "m", "Hi", "--speed", speed));
        assertTrue(e.getMessage().contains("--speed"));
    }

    @Test
    void errorsNameUnknownOptionsAndMissingValues() {
        assertTrue(
                assertThrows(Options.Usage.class, () -> Options.parse("chat", "--wat"))
                        .getMessage()
                        .contains("Unknown option: --wat"));
        assertTrue(
                assertThrows(Options.Usage.class, () -> Options.parse("chat", "--model"))
                        .getMessage()
                        .contains("Missing argument for option --model"));
        assertTrue(
                assertThrows(
                                Options.Usage.class,
                                () -> Options.parse("chat", "-m", "m", "--top-k", "many"))
                        .getMessage()
                        .contains("--top-k"));
        assertThrows(
                Options.Usage.class, () -> Options.parse("server", "-m", "m", "--task", "chat"));
    }

    @Test
    void requestSettingsFlowToExistingEngineTypes() {
        Options o =
                Options.parse(
                        "instruct",
                        "-m",
                        "m",
                        "hi",
                        "--max-reasoning-tokens",
                        "128",
                        "--reasoning-cutoff-message",
                        "Enough.",
                        "-n",
                        "512",
                        "--think",
                        "off");
        var request =
                Requests.of(List.of(Message.user("hi")), new Sampling(0f, 1f, 0, 0f, null), o);
        assertEquals(128, request.maxReasoningTokens());
        assertEquals("Enough.", request.reasoningCutoffMessage());
        assertEquals(512, request.maxOutputTokens());
        assertFalse(request.thinking());
        assertTrue(request.tools().isEmpty());
        assertTrue(request.stops().isEmpty());
        assertNull(request.grammar());
    }

    @Test
    void resolutionIsAnExplicitStep(@TempDir Path dir) throws Exception {
        Path file = Files.writeString(dir.resolve("model.gguf"), "not loaded by the resolver");
        Options o = Options.parse("chat", "-m", file.toString());
        assertEquals(file, o.resolve(ModelStore.of(dir.resolve("cache"))).model());
    }

    @Test
    void bothRuntimeSettingsLandBeforeEitherIsInitialized(@TempDir Path dir) throws Exception {
        Path output = dir.resolve("probe.txt");
        var command = CliFixtures.javaCommand();
        command.addAll(
                List.of(
                        "-cp",
                        System.getProperty("java.class.path"),
                        RuntimeProbe.class.getName()));
        Process p =
                new ProcessBuilder(command)
                        .redirectOutput(output.toFile())
                        .redirectError(ProcessBuilder.Redirect.INHERIT)
                        .start();
        try {
            assertTrue(p.waitFor(20, TimeUnit.SECONDS));
            assertEquals(0, p.exitValue());
            assertEquals("3:17", Files.readString(output).strip());
        } finally {
            p.destroyForcibly();
        }
    }

    public static class RuntimeProbe {
        public static void main(String[] args) throws Exception {
            Options.parse("-m", "m", "--threads", "3", "chat", "--batch-capacity", "17")
                    .configureRuntime();
            System.out.println(RuntimeFlags.THREADS + ":" + RuntimeFlags.BATCH_CAPACITY);
        }
    }

    @Test
    void existingMakefileInvocationStillParses() throws Exception {
        Path makefile = Path.of("..", "Makefile");
        int found = 0;
        for (String line : Files.readAllLines(makefile)) {
            int start = line.indexOf("-jar $(JAR_FILE)");
            if (start < 0 || !line.contains("--model")) continue;
            var tokens = new java.util.ArrayList<String>();
            var matcher =
                    java.util.regex.Pattern.compile("\"([^\"]*)\"|(\\S+)")
                            .matcher(
                                    line.substring(start + "-jar $(JAR_FILE)".length())
                                            .replace("$(MODEL)", "model.gguf"));
            while (matcher.find())
                tokens.add(matcher.group(1) != null ? matcher.group(1) : matcher.group(2));
            assertEquals("instruct", Options.parse(tokens.toArray(String[]::new)).command);
            found++;
        }
        assertTrue(found > 0);
    }
}
