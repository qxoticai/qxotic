package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.server.ServerConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The command line's own rules - the ones whose failure prints usage and exits 1, as opposed to
 * {@code Validation}'s, whose failure becomes a 400.
 */
final class OptionsTest {

    private static Options options(boolean server, boolean noGrammar) {
        return new Options(
                Path.of("model.gguf"),
                null,
                "hi",
                null,
                false,
                server,
                "127.0.0.1",
                0,
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
                noGrammar,
                null,
                false,
                null);
    }

    /**
     * --no-grammar refuses requests carrying a grammar, and only the HTTP API has requests. It used
     * to be accepted in chat and instruct mode and do nothing whatsoever.
     */
    @Test
    void noGrammarIsRejectedOutsideServerMode() {
        options(true, true);
        assertThrows(IllegalArgumentException.class, () -> options(false, true));
    }

    /** A prompt is required unless something else will supply one. */
    @Test
    void instructModeNeedsAPrompt() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new Options(
                                Path.of("model.gguf"),
                                null,
                                null,
                                null,
                                false,
                                false,
                                "127.0.0.1",
                                0,
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
                                false,
                                null,
                                false,
                                null));
    }

    @Test
    void aTransposedTemperatureAndTopPIsRejectedBeforeTheModelLoads() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new Options(
                                Path.of("model.gguf"),
                                null,
                                "hi",
                                null,
                                false,
                                false,
                                "127.0.0.1",
                                0,
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
                                false,
                                null,
                                false,
                                null));
    }

    /** Unset flags become the model's recommendations, and the projection carries them through. */
    @Test
    void toServerConfigResolvesSamplingAndNamesTheModel() {
        ServerConfig config =
                options(true, false).toServerConfig(LoadedModel.SamplingDefaults.NONE);
        assertEquals("model.gguf", config.modelName());
        assertEquals(0.8f, config.defaults().sampling().temperature());
        assertEquals(128, config.defaults().maxOutputTokens());
        assertEquals(ServerConfig.Limits.DEFAULTS.threads(), config.limits().threads());
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

    @Test
    void anUnknownFlagIsRefusedByName(@TempDir Path dir) throws IOException {
        String m = model(dir).toString();
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Options.parse(new String[] {"-m", m, "--frobnicate", "on"}));
        assertTrue(failure.getMessage().contains("--frobnicate"), failure.getMessage());
    }

    /** --no-grammar is the one flag that lands in Limits, because a request cannot lift it. */
    @Test
    void noGrammarBecomesALimit() {
        assertEquals(
                false,
                options(true, true)
                        .toServerConfig(LoadedModel.SamplingDefaults.NONE)
                        .limits()
                        .grammar());
    }
}
