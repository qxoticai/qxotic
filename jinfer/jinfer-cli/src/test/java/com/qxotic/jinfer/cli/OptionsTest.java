package com.qxotic.jinfer.cli;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.server.ServerConfig;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

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
        assertEquals(128, config.defaults().maxTokens());
        assertEquals(ServerConfig.Limits.DEFAULTS.threads(), config.limits().threads());
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
