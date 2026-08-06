package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.Server;
import com.qxotic.jinfer.server.ServerConfig;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.time.Duration;
import java.util.Locale;

/**
 * The parsed command line: every flag, exactly as typed, with the four sampling knobs still
 * nullable because "the user said nothing" is information the model's own recommendations need.
 *
 * <p>This is a CLI type, not a server one. It is wide because a command line is wide; what the
 * server needs is the narrow projection {@link #toServerConfig} builds, and nothing downstream of
 * that ever sees {@code --colors} or {@code --chat}. The two also validate different things: the
 * checks here exist to produce a good message next to a usage block, while {@link ServerConfig}
 * validates its own contract for any caller.
 *
 * @param maxTokens the CLI's budget flag. In server mode it becomes the DEFAULT completion budget
 *     for a request that omits {@code max_tokens}; in chat and instruct mode it is a total context
 *     cap, which is a second meaning the flag should not have (see the note in Main's usage)
 */
public record Options(
        Path modelPath,
        java.util.Map<String, Path> companions,
        String prompt,
        String systemPrompt,
        boolean interactive,
        boolean server,
        String host,
        int port,
        Float temperature,
        Float topp,
        Integer topk,
        Float minp,
        Long seed,
        int maxTokens,
        boolean stream,
        boolean echo,
        boolean think,
        boolean thinkInline,
        boolean colors,
        boolean rawPrompt,
        boolean noGrammar,
        Path promptCache,
        boolean promptCacheReadOnly,
        ServerConfig.Limits limits) {

    public Options {
        // never null and never the caller's copy: every reader can iterate it without a guard, and
        // "no companions" and "an empty map" stop being two states that behave differently
        companions = companions == null ? java.util.Map.of() : java.util.Map.copyOf(companions);
        limits = limits == null ? ServerConfig.Limits.DEFAULTS : limits;
        require(modelPath != null, "Missing argument: --model <path> is required");
        require(
                server || interactive || prompt != null,
                "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is"
                        + " the sky blue?\"");
        require(
                temperature == null || 0 <= temperature,
                "Invalid argument: --temperature must be non-negative");
        require(
                topp == null || (0 < topp && topp <= 1),
                "Invalid argument: --top-p must be within (0, 1] (1 disables it)");
        require(
                topk == null || topk >= 0,
                "Invalid argument: --top-k must be non-negative (0 disables it)");
        require(
                minp == null || (0 <= minp && minp <= 1),
                "Invalid argument: --min-p must be within [0, 1]");
        // checked here, before InetSocketAddress can say "port out of range" without saying which
        // flag or how to fix it
        require(0 <= port && port <= 65535, "Invalid argument: --port must be within [0, 65535]");
        // the only thing --no-grammar does is refuse requests that ask for a grammar, and only
        // the HTTP API has requests. Accepting it elsewhere made it a flag that did nothing.
        require(
                !noGrammar || server,
                "Invalid argument: --no-grammar applies to --server (it refuses requests carrying"
                        + " grammar or response_format); there is nothing to refuse in chat or"
                        + " instruct mode");
    }

    /**
     * The sampling stack these flags describe, over the model's own recommendations: an explicit
     * flag wins, then the container's {@code general.sampling.*}, then the engine baseline.
     */
    public Sampling sampling(LoadedModel.SamplingDefaults defaults) {
        return defaults.resolve(temperature, topp, topk, minp, seed);
    }

    /**
     * The narrow projection {@link Server#start} takes. This is the one place flags become server
     * configuration, which is why the server can promise it reads nothing else.
     */
    public ServerConfig toServerConfig(LoadedModel.SamplingDefaults defaults) {
        return new ServerConfig(
                modelPath.getFileName().toString(),
                new InetSocketAddress(host, port),
                new ServerConfig.Defaults(sampling(defaults), maxTokens, think, rawPrompt),
                limits.withGrammar(!noGrammar),
                PromptCache.Options.DEFAULTS.withCatalog(promptCache, promptCacheReadOnly));
    }

    /**
     * Rejects a bad COMMAND LINE. The message is printed next to the usage block and the process
     * exits 1, which is why it names flags; jinfer-server's {@code Validation.require} is its
     * request-side counterpart, whose messages travel to a client in a 400 and must not. Two copies
     * of four lines, because the two contracts are genuinely different - and because this module
     * cannot see that one, which is the boundary doing its job.
     */
    public static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }

    public static boolean parseBooleanOption(String optionName, String value) {
        return switch (value.toLowerCase(Locale.ROOT)) {
            case "true", "on" -> true;
            case "false", "off" -> false;
            default -> {
                require(
                        false,
                        "Invalid argument for %s: expected true|false|on|off, got %s",
                        optionName,
                        value);
                yield false;
            }
        };
    }

    /** Seconds on the command line, a {@link Duration} everywhere after it. */
    public static Duration seconds(String optionName, String value) {
        long seconds;
        try {
            seconds = Long.parseLong(value);
        } catch (NumberFormatException e) {
            require(false, "Invalid argument for %s: expected seconds, got %s", optionName, value);
            return Duration.ZERO;
        }
        require(seconds >= 0, "Invalid argument for %s: must be non-negative", optionName);
        return Duration.ofSeconds(seconds);
    }

    public static boolean supportsAnsiColors(String colorMode) {
        return switch (colorMode) {
            case "on" -> true;
            case "off" -> false;
            case "auto" -> {
                if (System.console() == null) {
                    yield false;
                }
                String noColor = System.getenv("NO_COLOR");
                if (noColor != null) {
                    yield false;
                }
                String term = System.getenv("TERM");
                yield term == null || !"dumb".equalsIgnoreCase(term);
            }
            default -> false;
        };
    }
}
