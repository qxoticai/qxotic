package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/** The command line's own rules - the ones whose failure prints usage and exits 1. */
final class OptionsTest {

    private static Options options(String prompt) {
        return new Options(
                Path.of("model.gguf"),
                null,
                null,
                prompt,
                null,
                false,
                null,
                null,
                null,
                null,
                null,
                128,
                4096,
                true,
                false,
                true,
                false,
                false,
                false,
                null,
                false,
                4);
    }

    @Test
    void helpSaysWhatThinkOffDoes() {
        // --think off disables reasoning at the prompt (the model answers directly); the help
        // used to promise a display filter over thoughts the model "still generates"
        java.io.ByteArrayOutputStream out = new java.io.ByteArrayOutputStream();
        Options.printUsage(new java.io.PrintStream(out, true));
        String help = out.toString();
        assertTrue(help.contains("off: do not reason"), help);
        assertFalse(help.contains("still generates"), help);
    }

    /**
     * --with media=x.gguf and no --model used to reach the companion header read with a null model
     * path and print the raw NPE; the remedy is the flag's name, before any resolution.
     */
    @Test
    void aCompanionWithoutAModelNamesTheModelFlag(@TempDir Path dir) throws IOException {
        Path mmproj = Files.createFile(dir.resolve("mmproj.gguf"));
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Options.parse(new String[] {"--with", "media=" + mmproj}));
        assertTrue(e.getMessage().contains("--model"), e.getMessage());
    }

    @Test
    void cacheInChatModeIsRefused(@TempDir Path dir) throws IOException {
        // a flag that does nothing is refused, not ignored - the chat loop keeps its own state
        // and never consults the prompt cache
        Path model = Files.createFile(dir.resolve("m.gguf"));
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Options.parse(
                                        new String[] {
                                            "--model",
                                            model.toString(),
                                            "--chat",
                                            "--cache",
                                            dir.resolve("c.jkv").toString()
                                        }));
        assertTrue(e.getMessage().contains("--cache"), e.getMessage());
    }

    /** A prompt is required unless something else will supply one. */
    @Test
    void instructModeNeedsAPrompt() {
        assertThrows(IllegalArgumentException.class, () -> options(null));
    }

    @Test
    void reasoningBudgetFlagsParse(@TempDir Path dir) throws IOException {
        Path model = Files.createFile(dir.resolve("m.gguf"));
        Options options =
                Options.parse(
                        new String[] {
                            "--model", model.toString(),
                            "-p", "hi",
                            "--reasoning-budget", "128",
                            "--reasoning-budget-message", "... Let me wrap up."
                        });
        assertEquals(128, options.reasoningBudget());
        assertEquals("... Let me wrap up.", options.reasoningBudgetMessage());
    }

    @Test
    void reasoningBudgetBelowMinusOneNamesTheFlag(@TempDir Path dir) throws IOException {
        Path model = Files.createFile(dir.resolve("m.gguf"));
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Options.parse(
                                        new String[] {
                                            "--model", model.toString(),
                                            "-p", "hi",
                                            "--reasoning-budget", "-2"
                                        }));
        assertTrue(e.getMessage().contains("--reasoning-budget"), e.getMessage());
    }

    @Test
    void reasoningFlagsFlowIntoTheEngineRequest(@TempDir Path dir) throws IOException {
        Path model = Files.createFile(dir.resolve("m.gguf"));
        Options options =
                Options.parse(
                        new String[] {
                            "--model", model.toString(),
                            "-p", "hi",
                            "--reasoning-budget", "128",
                            "--reasoning-budget-message", "... Let me wrap up."
                        });
        var request =
                Requests.of(
                        java.util.List.of(com.qxotic.jinfer.chat.Message.user("hi")),
                        new com.qxotic.jinfer.llm.Sampling(0f, 1f, 0, 0f, null),
                        options);
        assertEquals(128, request.reasoningMaxTokens());
        assertEquals("... Let me wrap up.", request.reasoningMessage());
    }

    @Test
    void aTransposedTemperatureAndTopPIsRejectedBeforeTheModelLoads() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new Options(
                                        Path.of("model.gguf"),
                                        null,
                                        null,
                                        "hi",
                                        null,
                                        false,
                                        0.95f,
                                        1.7f,
                                        null,
                                        null,
                                        null,
                                        128,
                                        4096,
                                        true,
                                        false,
                                        true,
                                        false,
                                        false,
                                        false,
                                        null,
                                        false,
                                        4));
        assertTrue(e.getMessage().contains("--top-p"), e.getMessage());
    }

    /** Unset flags become the model's recommendations, then the engine baseline. */
    @Test
    void unsetSamplingResolvesToTheBaseline() {
        var sampling = options("hi").sampling(LoadedModel.SamplingDefaults.NONE);
        assertEquals(0.8f, sampling.temperature());
        assertEquals(0.95f, sampling.topP());
        assertEquals(40, sampling.topK());
    }

    // ---- parse: argv to Options ----

    private static Path model(Path dir) throws IOException {
        return Files.writeString(dir.resolve("model.gguf"), "not really a model");
    }

    @Test
    void parseReadsFlagsInBothSpellings(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        Options options =
                Options.parse(
                        new String[] {
                            "-m", m, "-p", "hi", "--temp", "0.5", "--context-capacity=512"
                        });
        assertEquals(0.5f, options.temperature());
        assertEquals(512, options.contextCapacity(), "--flag=value works like --flag value");
    }

    @Test
    void zeroContextCapacitySelectsTheModelMaximum(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        Options options =
                Options.parse(new String[] {"-m", m, "-p", "hi", "--context-capacity", "0"});
        assertEquals(0, options.contextCapacity());
    }

    @Test
    void negativeContextCapacityIsRejectedByName(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Options.parse(
                                        new String[] {
                                            "-m", m, "-p", "hi", "--context-capacity", "-1"
                                        }));
        assertTrue(failure.getMessage().contains("--context-capacity"), failure.getMessage());
    }

    /**
     * A bad number names its flag; 'For input string: \"x\"' named neither flag nor expectation.
     */
    @Test
    void aBadNumberNamesItsFlag(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Options.parse(new String[] {"-m", m, "-p", "hi", "--top-k", "many"}));
        assertTrue(failure.getMessage().contains("--top-k"), failure.getMessage());
        assertTrue(failure.getMessage().contains("many"), failure.getMessage());
    }

    /** model= and tokenizer= are RESERVED --with roles: they route to their own seams. */
    @Test
    void reservedWithRolesRouteToTheirSeams(@TempDir Path dir) throws IOException {
        Path m = model(dir);
        Path t = Files.writeString(dir.resolve("other.gguf"), "another model");
        Options options =
                Options.parse(
                        new String[] {
                            "--with", "model=" + m, "--with", "tokenizer=" + t, "-p", "hi"
                        });
        assertEquals(m, options.modelPath(), "--with model= is -m by another spelling");
        assertEquals(t, options.tokenizerPath());
        assertEquals(0, options.companions().size(), "reserved roles are not companions");
    }

    @Test
    void writableCacheWithRawPromptIsRefused(@TempDir Path dir) throws IOException {
        // definePrompt goes through the native codec's conversation encoding, which a raw prompt
        // deliberately bypasses - accepting the flag would silently never append
        Path model = Files.createFile(dir.resolve("m.gguf"));
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                Options.parse(
                                        new String[] {
                                            "--model",
                                            model.toString(),
                                            "--raw-prompt",
                                            "-p",
                                            "hi",
                                            "--cache",
                                            dir.resolve("c.jkv").toString()
                                        }));
        assertTrue(e.getMessage().contains("--cache-ro"), e.getMessage());
        // read-only is fine: the raw batch is served as-is
        Files.writeString(dir.resolve("c.jkv"), "");
        Options ok =
                Options.parse(
                        new String[] {
                            "--model",
                            model.toString(),
                            "--raw-prompt",
                            "-p",
                            "hi",
                            "--cache-ro",
                            dir.resolve("c.jkv").toString()
                        });
        assertTrue(ok.promptCacheReadOnly());
    }

    @Test
    void anUnknownFlagIsRefusedByName(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Options.parse(new String[] {"-m", m, "--frobnicate", "on"}));
        assertTrue(failure.getMessage().contains("--frobnicate"), failure.getMessage());
    }

    @Test
    void serverModeNeedsNoPromptAndOwnsOnlyTransportPolicy(@TempDir Path dir) throws IOException {
        String model = model(dir).toString();
        Options options =
                Options.parse(
                        new String[] {
                            "-m",
                            model,
                            "--server",
                            "--port",
                            "0",
                            "--queue-capacity",
                            "2",
                            "--speculation-depth",
                            "6"
                        });
        assertTrue(options.server());
        assertEquals(6, options.speculationDepth(), "MTP depth stays an engine option");
        assertEquals(2, options.limits().queueCapacity());
        assertEquals(
                0,
                options.serverConfig(options.sampling(LoadedModel.SamplingDefaults.NONE))
                        .bind()
                        .getPort());
    }

    @Test
    void publicBindRequiresAuthentication(@TempDir Path dir) throws IOException {
        String model = model(dir).toString();
        assertThrows(
                IllegalArgumentException.class,
                () -> Options.parse(new String[] {"-m", model, "--server", "--host", "0.0.0.0"}));
        Options secured =
                Options.parse(
                        new String[] {
                            "-m", model, "--server", "--host", "0.0.0.0", "--api-key", "secret"
                        });
        assertEquals("secret", secured.apiKey());
    }
}
