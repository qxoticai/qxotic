package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import java.nio.file.Path;
import java.util.Locale;

public record LLMOptions(
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
        boolean promptCacheReadOnly) {

    public LLMOptions {
        // never null and never the caller's copy: every reader can iterate it without a guard, and
        // "no companions" and "an empty map" stop being two states that behave differently
        companions = companions == null ? java.util.Map.of() : java.util.Map.copyOf(companions);
        require(modelPath != null, "Missing argument: --model <path> is required");
        require(
                server || interactive || prompt != null,
                "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is"
                        + " the sky blue?\"");
        require(
                temperature == null || 0 <= temperature,
                "Invalid argument: --temperature must be non-negative");
        require(
                topp == null || (0 <= topp && topp <= 1),
                "Invalid argument: --top-p must be within [0, 1]");
        require(
                topk == null || topk >= 0,
                "Invalid argument: --top-k must be non-negative (0 disables it)");
        require(
                minp == null || (0 <= minp && minp <= 1),
                "Invalid argument: --min-p must be within [0, 1]");
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
     * Fills unset sampling flags from the container's recommendations ({@code general.sampling.*}
     * via {@link com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults}), then the shared engine
     * baseline. An explicit CLI flag always wins. Called once, right after the model loads;
     * everything downstream reads resolved, non-null values.
     */
    public LLMOptions withResolvedSampling(
            com.qxotic.jinfer.chat.LoadedModel.SamplingDefaults defaults) {
        com.qxotic.jinfer.llm.Sampling resolved =
                defaults.resolve(temperature, topp, topk, minp, seed);
        return new LLMOptions(
                modelPath,
                companions,
                prompt,
                systemPrompt,
                interactive,
                server,
                host,
                port,
                resolved.temperature(),
                resolved.topP(),
                resolved.topK(),
                resolved.minP(),
                seed,
                maxTokens,
                stream,
                echo,
                think,
                thinkInline,
                colors,
                rawPrompt,
                noGrammar,
                promptCache,
                promptCacheReadOnly);
    }

    /**
     * The sampling stack these options describe. Valid only AFTER {@link #withResolvedSampling}:
     * before it the four knobs are still nullable, and {@link com.qxotic.jinfer.llm.Sampling} takes
     * values, not maybes.
     */
    public com.qxotic.jinfer.llm.Sampling sampling() {
        return new com.qxotic.jinfer.llm.Sampling(temperature, topp, topk, minp, seed);
    }

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
