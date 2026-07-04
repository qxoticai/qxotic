package com.qxotic.jinfer;

import java.nio.file.Path;
import java.util.List;
import java.util.Locale;

public record LLMOptions(Path modelPath, String prompt, String suffix, String systemPrompt,
                          boolean interactive, boolean server, String host, int port,
                          float temperature, float topp, long seed, int maxTokens, boolean stream,
                          boolean echo, boolean think, boolean thinkInline, boolean colors,
                          boolean keepPastThinking, boolean rawPrompt, List<String> warmPrompts,
                          boolean noGrammar, Path sealedPrompt) {

    public LLMOptions {
        require(modelPath != null, "Missing argument: --model <path> is required");
        require(server || interactive || prompt != null,
                "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
        require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
        require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
        require(0 <= port && port <= 65535, "Invalid argument: --port must be within [0, 65535]");
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
                require(false, "Invalid argument for %s: expected true|false|on|off, got %s", optionName, value);
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
