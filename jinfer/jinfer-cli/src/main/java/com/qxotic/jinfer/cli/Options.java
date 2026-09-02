package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.ServerConfig;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;

/**
 * The parsed command line: every flag, exactly as typed, with the four sampling knobs still
 * nullable because "the user said nothing" is information the model's own recommendations need.
 * {@link #parse} is the only way argv becomes one of these, and {@link #printUsage} documents the
 * same flags, so a new flag and its help line are added in the same file or noticed missing.
 *
 * @param maxOutputTokens tokens GENERATED per turn, -1 = as many as the context allows
 * @param contextCapacity the size of a session's state, and the ceiling on every one-shot; {@code
 *     null} (not given) is the engine's bounded default, min(4096, the model's context length);
 *     {@code 0} uses the model's context length; positive values above it are refused at load
 * @param threads compute threads ({@code --threads}): the one pool every kernel and jam backend
 *     runs on, the same knob as {@code -Djinfer.threads}; {@code null} (not given) is one per
 *     physical core. Server admission is {@code --concurrency}, a different thing.
 */
public record Options(
        Path modelPath,
        Map<String, Path> companions,
        Path tokenizerPath,
        String prompt,
        String systemPrompt,
        boolean interactive,
        Float temperature,
        Float topp,
        Integer topk,
        Float minp,
        Long seed,
        int maxOutputTokens,
        Integer contextCapacity,
        boolean stream,
        boolean echo,
        boolean think,
        boolean thinkInline,
        Integer reasoningBudget,
        String reasoningBudgetMessage,
        boolean colors,
        boolean rawPrompt,
        Path promptCache,
        boolean promptCacheReadOnly,
        int speculationDepth,
        boolean server,
        String host,
        int port,
        String apiKey,
        Set<String> allowedOrigins,
        boolean noGrammar,
        ServerConfig.Limits limits,
        Integer threads) {

    /** The default bind port for {@code --server} (overridable with {@code --port}). */
    public static final int DEFAULT_PORT = 54154;

    public Options {
        // never null and never the caller's copy: every reader can iterate it without a guard, and
        // "no companions" and "an empty map" stop being two states that behave differently
        companions = companions == null ? Map.of() : Map.copyOf(companions);
        host = host == null ? "127.0.0.1" : host;
        allowedOrigins = allowedOrigins == null ? Set.of("*") : Set.copyOf(allowedOrigins);
        limits = limits == null ? ServerConfig.Limits.DEFAULTS : limits;
        require(modelPath != null, "Missing argument: --model <path> is required");
        require(
                server || interactive || prompt != null,
                "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is"
                        + " the sky blue?\"");
        require(
                temperature == null || 0 <= temperature,
                "Invalid argument: --temp must be non-negative");
        require(threads == null || threads >= 1, "Invalid argument: --threads must be at least 1");
        require(
                topp == null || (0 < topp && topp <= 1),
                "Invalid argument: --top-p must be within (0, 1] (1 disables it)");
        require(
                topk == null || topk >= 0,
                "Invalid argument: --top-k must be non-negative (0 disables it)");
        require(
                minp == null || (0 <= minp && minp <= 1),
                "Invalid argument: --min-p must be within [0, 1]");
        require(
                contextCapacity == null || contextCapacity >= 0,
                "Invalid argument: --context-capacity must be non-negative"
                        + " (0 uses the model maximum)");
        require(0 <= port && port <= 65535, "Invalid argument: --port must be within [0, 65535]");
        require(
                maxOutputTokens >= -1,
                "Invalid argument: --max-output-tokens must be -1 (fill the context) or"
                        + " non-negative");
        // the chat loop keeps its own state and never consults the prompt cache; accepting the
        // flag there would make it do nothing
        require(
                promptCache == null || !interactive || server,
                "Invalid argument: --cache/--cache-ro apply to instruct and server modes; the"
                        + " chat loop keeps its own state, so there is nothing for the cache to"
                        + " serve");
        // defining a cache entry goes through the native codec's conversation encoding, which a
        // raw prompt deliberately bypasses. Read-only is fine: the raw batch is served as-is
        require(
                promptCache == null || promptCacheReadOnly || !rawPrompt,
                "Invalid argument: --cache with --raw-prompt cannot append (there is no"
                        + " conversation to define) - use --cache-ro to serve an existing cache");
        require(
                0 <= speculationDepth && speculationDepth <= 8,
                "Invalid argument: --speculation-depth must be within [0, 8] (0 disables it)");
        require(!noGrammar || server, "Invalid argument: --no-grammar applies only to --server");
        if (server) {
            InetSocketAddress address = new InetSocketAddress(host, port);
            require(
                    !address.isUnresolved(),
                    "Invalid argument: --host " + host + " does not resolve");
            require(
                    address.getAddress().isLoopbackAddress() || apiKey != null,
                    "Invalid argument: a non-loopback --host requires --api-key");
        }
        // a raw prompt is the model's input verbatim: nothing that the chat template would
        // have framed can apply, and the chat loop is all template
        if (rawPrompt) {
            require(!interactive, "Invalid argument: --raw-prompt applies to --prompt, not --chat");
            require(
                    systemPrompt == null
                            && reasoningBudget == null
                            && reasoningBudgetMessage == null,
                    "Invalid argument: --raw-prompt bypasses the chat template, so --system-prompt"
                            + " and --reasoning-budget* cannot apply");
        }
    }

    /** Compatibility constructor for chat/instruct callers and tests. */
    public Options(
            Path modelPath,
            Map<String, Path> companions,
            Path tokenizerPath,
            String prompt,
            String systemPrompt,
            boolean interactive,
            Float temperature,
            Float topp,
            Integer topk,
            Float minp,
            Long seed,
            int maxOutputTokens,
            Integer contextCapacity,
            boolean stream,
            boolean echo,
            boolean think,
            boolean thinkInline,
            boolean colors,
            boolean rawPrompt,
            Path promptCache,
            boolean promptCacheReadOnly,
            int speculationDepth) {
        this(
                modelPath,
                companions,
                tokenizerPath,
                prompt,
                systemPrompt,
                interactive,
                temperature,
                topp,
                topk,
                minp,
                seed,
                maxOutputTokens,
                contextCapacity,
                stream,
                echo,
                think,
                thinkInline,
                null,
                null,
                colors,
                rawPrompt,
                promptCache,
                promptCacheReadOnly,
                speculationDepth,
                false,
                "127.0.0.1",
                DEFAULT_PORT,
                null,
                Set.of("*"),
                false,
                ServerConfig.Limits.DEFAULTS,
                null);
    }

    ServerConfig serverConfig(Sampling sampling) {
        return new ServerConfig(
                new InetSocketAddress(host, port),
                new ServerConfig.Defaults(
                        sampling,
                        maxOutputTokens,
                        think,
                        rawPrompt,
                        reasoningBudget,
                        reasoningBudgetMessage),
                limits.withGrammar(!noGrammar),
                new ServerConfig.Access(apiKey, allowedOrigins));
    }

    /**
     * Refuses a capacity larger than the model was trained for - which needs the model, so it is
     * checked once, right after the load, rather than in the compact constructor with the flags
     * that stand on their own.
     */

    /**
     * The sampling stack these flags describe, over the model's own recommendations: an explicit
     * flag wins, then the container's {@code general.sampling.*}, then the engine baseline.
     */
    public Sampling sampling(LoadedModel.SamplingDefaults defaults) {
        return defaults.resolve(temperature, topp, topk, minp, seed);
    }

    /**
     * Rejects a bad COMMAND LINE. The message is printed next to the usage block and the process
     * exits 1, which is why it names flags.
     */
    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }

    static boolean parseBooleanOption(String optionName, String value) {
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

    // number parsing that names the flag: a raw NumberFormatException surfaces as
    // 'For input string: "abc"', which tells the user neither which flag nor what it expected
    private static int parseInt(String optionName, String value) {
        try {
            return Integer.parseInt(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid argument for %s: expected an integer, got %s"
                            .formatted(optionName, value));
        }
    }

    private static long parseLong(String optionName, String value) {
        try {
            return Long.parseLong(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid argument for %s: expected an integer, got %s"
                            .formatted(optionName, value));
        }
    }

    private static float parseFloat(String optionName, String value) {
        try {
            return Float.parseFloat(value);
        } catch (NumberFormatException e) {
            throw new IllegalArgumentException(
                    "Invalid argument for %s: expected a number, got %s"
                            .formatted(optionName, value));
        }
    }

    private static Duration parseSeconds(String optionName, String value) {
        long seconds = parseLong(optionName, value);
        require(seconds >= 0, "Invalid argument for %s: must be non-negative", optionName);
        return Duration.ofSeconds(seconds);
    }

    static boolean supportsAnsiColors(String colorMode) {
        return switch (colorMode) {
            case "on" -> true;
            case "off" -> false;
            case "auto" -> {
                // isTerminal(), NOT console() != null: since JDK 22 a Console is handed out even
                // when output is redirected, and ANSI escapes in a piped file are corruption
                if (System.console() == null || !System.console().isTerminal()) {
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

    /** A model ref that could not be resolved: the cause already says what to do about it. */
    static final class ResolveFailure extends RuntimeException {
        ResolveFailure(RuntimeException cause) {
            super(cause.getMessage(), cause); // the cause's own words, not its class name
        }
    }

    /** The remedy a failure carries, unwrapping the plumbing around it. */
    static String rootMessage(Throwable failure) {
        Throwable root = failure;
        while (root.getMessage() == null && root.getCause() != null) {
            root = root.getCause();
        }
        return root.getMessage();
    }

    static Options parse(String[] args) {
        String prompt = null;
        String systemPrompt = null;
        Float temperature = null; // unset = the model's recommended value, else 0.8
        Float topp = null; // unset = the model's recommended value, else 0.95
        Integer topk = null; // unset = the model's recommended value, else 40
        Float minp = null; // unset = the model's recommended value, else 0.05
        // paths or hub refs; resolved (and downloaded, if needed) once parsing has succeeded
        String modelRef = null;
        String tokenizerRef = null;
        // capability -> path or ref; resolved once parsing has succeeded
        Map<String, String> companionRefs = new LinkedHashMap<>();
        Long seed = null; // unset = a fresh random seed per request
        int maxOutputTokens = -1;
        Integer contextCapacity = null; // the engine's bounded default
        boolean interactive = false;
        boolean server = false;
        String host = "127.0.0.1";
        int port = DEFAULT_PORT;
        String apiKey = null;
        Set<String> allowedOrigins = new TreeSet<>();
        boolean noGrammar = false;
        Integer threads = null; // unset = one per physical core (RuntimeFlags)
        ServerConfig.Limits defaultLimits = ServerConfig.Limits.DEFAULTS;
        int concurrency = defaultLimits.threads();
        int queueCapacity = defaultLimits.queueCapacity();
        long maxBodyBytes = defaultLimits.maxBodyBytes();
        Duration writeTimeout = defaultLimits.writeTimeout();
        Duration requestTimeout = defaultLimits.requestTimeout();
        boolean stream = true;
        boolean echo = false;
        boolean think = true;
        boolean thinkInline = false;
        Integer reasoningBudget = null;
        String reasoningBudgetMessage = null;
        String colorMode = "auto";
        boolean rawPrompt = false;
        Path promptCache = null;
        boolean promptCacheReadOnly = false;
        int speculationDepth = 4;

        for (int i = 0; i < args.length; i++) {
            String optionName = args[i];
            require(optionName.startsWith("-"), "Invalid option %s", optionName);
            switch (optionName) {
                case "--interactive", "--chat", "-i" -> interactive = true;
                case "--instruct" -> interactive = false;
                case "--server" -> server = true;
                case "--raw-prompt" -> rawPrompt = true;
                case "--no-grammar" -> noGrammar = true;
                case "--help", "-h" -> {
                    printUsage(System.out);
                    System.exit(0);
                }
                default -> {
                    String nextArg;
                    if (optionName.contains("=")) {
                        String[] parts = optionName.split("=", 2);
                        optionName = parts[0];
                        nextArg = parts[1];
                    } else {
                        require(i + 1 < args.length, "Missing argument for option %s", optionName);
                        nextArg = args[i + 1];
                        i += 1;
                    }
                    switch (optionName) {
                        case "--prompt", "-p" -> prompt = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--temp" -> temperature = parseFloat(optionName, nextArg);
                        case "--top-p" -> topp = parseFloat(optionName, nextArg);
                        case "--top-k" -> topk = parseInt(optionName, nextArg);
                        case "--min-p" -> minp = parseFloat(optionName, nextArg);
                        case "--model", "-m" -> modelRef = nextArg;
                        case "--mmproj" -> companionRefs.put("media", nextArg);
                        case "--with" -> {
                            int eq = nextArg.indexOf('=');
                            require(
                                    eq > 0 && eq < nextArg.length() - 1,
                                    "--with takes <role>=<path|ref>, got %s",
                                    nextArg);
                            String role = nextArg.substring(0, eq);
                            String value = nextArg.substring(eq + 1);
                            // ONE attachment syntax; two roles are RESERVED and route to their
                            // own seams (the model anchors the load, the tokenizer is a typed
                            // load argument), everything else is a companion capability that the
                            // model's port validates
                            switch (role) {
                                case "model" -> modelRef = value;
                                case "tokenizer" -> tokenizerRef = value;
                                default -> companionRefs.put(role, value);
                            }
                        }
                        case "--host" -> host = nextArg;
                        case "--port" -> port = parseInt(optionName, nextArg);
                        case "--api-key" -> apiKey = nextArg;
                        case "--cors-origin" -> allowedOrigins.add(nextArg);
                        // -t is llama.cpp's spelling, and jinfer-bench's
                        case "--threads", "-t" -> threads = parseInt(optionName, nextArg);
                        case "--concurrency" -> concurrency = parseInt(optionName, nextArg);
                        case "--queue-capacity" -> queueCapacity = parseInt(optionName, nextArg);
                        case "--max-body-mb" ->
                                maxBodyBytes = (long) parseInt(optionName, nextArg) << 20;
                        case "--write-timeout" -> writeTimeout = parseSeconds(optionName, nextArg);
                        case "--request-timeout" ->
                                requestTimeout = parseSeconds(optionName, nextArg);
                        case "--seed", "-s" -> seed = parseLong(optionName, nextArg);
                        // -n is llama.cpp's spelling for the same knob, and this CLI already
                        // honours its muscle memory for -m/-p/-c/-s/--temp
                        case "--max-output-tokens", "-n" ->
                                maxOutputTokens = parseInt(optionName, nextArg);
                        case "--context-capacity", "-c" ->
                                contextCapacity = parseInt(optionName, nextArg);
                        case "--speculation-depth" ->
                                speculationDepth = parseInt(optionName, nextArg);
                        case "--stream" -> stream = parseBooleanOption(optionName, nextArg);
                        case "--echo" -> echo = parseBooleanOption(optionName, nextArg);
                        case "--reasoning-budget" -> {
                            reasoningBudget = parseInt(optionName, nextArg);
                            require(
                                    reasoningBudget >= -1,
                                    "Invalid argument for %s: -1 (uncapped) or >= 0, got %s",
                                    optionName,
                                    nextArg);
                        }
                        case "--reasoning-budget-message" -> reasoningBudgetMessage = nextArg;
                        case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                        case "--cache" -> {
                            promptCache = Path.of(nextArg);
                            promptCacheReadOnly = false; // the later flag wins
                        }
                        case "--cache-ro" -> {
                            promptCache = Path.of(nextArg);
                            promptCacheReadOnly = true;
                        }
                        case "--think" -> {
                            String thinkMode = nextArg.toLowerCase(Locale.ROOT);
                            thinkInline = List.of("inline", "stdout").contains(thinkMode);
                            switch (thinkMode) {
                                case "on", "true", "inline", "stdout" -> think = true;
                                case "off", "false" -> think = false;
                                default ->
                                        require(
                                                false,
                                                "Invalid argument for %s: expected off|on|inline"
                                                        + " (or false|true|stdout), got %s",
                                                optionName,
                                                nextArg);
                            }
                        }
                        default -> require(false, "Unknown option: %s", optionName);
                    }
                }
            }
        }
        require(
                List.of("on", "off", "auto").contains(colorMode),
                "Invalid argument: --color must be one of on|off|auto");
        boolean color = supportsAnsiColors(colorMode);
        // BEFORE any resolution: a companion or tokenizer without a model would otherwise reach
        // the model-header read with a null path and die on the NPE instead of naming the flag
        require(modelRef != null, "Missing argument: --model <path> is required");
        // AFTER the loop, so --help and a bad flag never trigger a download first
        Path modelPath;
        Path tokenizerPath;
        Map<String, Path> companions = new LinkedHashMap<>();
        try {
            // EVERYTHING in one concurrent batch - a cold start pays the slowest download,
            // not the sum. The --with capability check needs the MODEL's header, so it runs the
            // moment the model lands; the trade is honest: a mistyped knob now costs a cached
            // extra file (almost always the very file the fixed knob would use), not gigabytes
            // of wasted WAITING. "auto" never reaches the resolver - its curated refusal below
            // beats "no such model file: 'auto'".
            List<String> wanted = new ArrayList<>();
            if (modelRef != null) wanted.add(modelRef);
            if (tokenizerRef != null) wanted.add(tokenizerRef);
            for (String value : companionRefs.values()) {
                if (!"auto".equals(value)) wanted.add(value);
            }
            List<Path> resolved = ModelStore.standard().resolveAll(wanted);
            int at = 0;
            modelPath = modelRef == null ? null : resolved.get(at++);
            tokenizerPath = tokenizerRef == null ? null : resolved.get(at++);
            if (!companionRefs.isEmpty()) {
                // ONE header read at most (none when the model is preloaded - AOT answers from
                // its baked header)
                Map<String, String> offered;
                try {
                    offered = AOT.companionFiles(modelPath);
                } catch (IOException e) {
                    throw new UncheckedIOException(e);
                }
                for (var w : companionRefs.entrySet()) {
                    requireCompanion(offered, w.getKey(), w.getValue());
                }
                for (String capability : companionRefs.keySet()) {
                    companions.put(capability, resolved.get(at++));
                }
            }
        } catch (RuntimeException e) {
            // a ref that cannot be resolved carries its own remedy (gated repo, unknown quant,
            // no such repository); the usage block would only bury it
            throw new ResolveFailure(e);
        }
        return new Options(
                modelPath,
                Map.copyOf(companions),
                tokenizerPath,
                prompt,
                systemPrompt,
                interactive,
                temperature,
                topp,
                topk,
                minp,
                seed,
                maxOutputTokens,
                contextCapacity,
                stream,
                echo,
                think,
                thinkInline,
                reasoningBudget,
                reasoningBudgetMessage,
                color,
                rawPrompt,
                promptCache,
                promptCacheReadOnly,
                speculationDepth,
                server,
                host,
                port,
                apiKey,
                allowedOrigins.isEmpty() ? Set.of("*") : allowedOrigins,
                noGrammar,
                new ServerConfig.Limits(
                        concurrency,
                        queueCapacity,
                        maxBodyBytes,
                        !noGrammar,
                        writeTimeout,
                        requestTimeout,
                        defaultLimits.shutdownTimeout()),
                threads);
    }

    /**
     * One companion, named EXPLICITLY. There is no "find it for me": a companion changes what the
     * model produces - a projector is in the media cache key - so which file it is belongs in the
     * command line where it can be read, reproduced and pinned.
     */
    private static void requireCompanion(
            Map<String, String> offered, String capability, String value) {
        String fileName = offered.get(capability);
        require(
                fileName != null,
                "this model has no '%s' capability. It offers: %s",
                capability,
                offered.isEmpty() ? "none" : new TreeSet<>(offered.keySet()));
        require(
                !"auto".equals(value),
                "name the '%s' file rather than 'auto' - it is usually called %s*, and browsing"
                        + " the model's repository shows which precisions it ships",
                capability,
                fileName);
    }

    static void printUsage(PrintStream out) {
        out.println("Usage:  java -jar jinfer.jar [options]");
        out.println("        JVM: --add-modules jdk.incubator.vector");
        out.println(
                "        java -jar jinfer.jar pull [--force] <ref>...  download models, print"
                        + " paths");
        out.println(
                "        java -jar jinfer.jar list                     cached models and sizes");
        out.println();
        out.println(
                "A remote model is a URL without the scheme: "
                        + " [<host>/]<owner>/<repo>[@rev][/path][:quant]");
        out.println(
                "  unsloth/gemma-4-E2B-it-GGUF:Q8_0             "
                        + " modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0");
        out.println(
                "Host is optional and defaults to hf.co; modelscope.cn is the other one. Quant"
                        + " defaults to Q4_K_M.");
        out.println("An existing file of that name wins, and anything else is a local file path.");
        out.println(
                "Cache: $JINFER_MODELS, else the platform cache dir. HF_TOKEN for gated repos,"
                        + " JINFER_OFFLINE=1 to never fetch.");
        out.println();
        out.println("Model:");
        out.println(
                "  --model, -m <path|ref>        required, a .gguf file or a remote model -"
                        + " downloaded on first use");
        out.println("  --mmproj <path|ref>           shorthand for --with media=<...>");
        out.println(
                "  --with <role>=<path|ref>      attach a file by role. model= (same as -m) and"
                        + " tokenizer= (use another GGUF's tokenizer; id-space checked) are"
                        + " reserved; any other role is a COMPANION capability of the"
                        + " architecture, e.g. gemma4: media (vision/audio encoders)");
        out.println();
        out.println("Mode (one of; default --instruct):");
        out.println("  --instruct                    answer --prompt once and exit");
        out.println("  --interactive, --chat, -i     chat in the terminal");
        out.println("  --server                      serve an OpenAI-compatible HTTP API");
        out.println();
        out.println("Prompt:");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
        out.println(
                "  --raw-prompt                  bypass chat template and tokenize --prompt"
                        + " directly (no system prompt, thinking or budget)");
        out.println(
                "  --think <off|on|inline>       on: reason, thoughts on stderr (default); off: do"
                        + " not reason, the model answers directly; inline: thoughts on stdout");
        out.println(
                "  --reasoning-budget <int>      cap the thinking span at N tokens (default: model"
                        + " policy, -1: uncapped)");
        out.println(
                "  --reasoning-budget-message <s>  forced as the model's own words when the budget"
                        + " runs out, e.g. \"... Let me wrap up.\" (default: a paragraph break)");
        out.println();
        out.println("Sampling (default: the model's recommended values):");
        out.println("  --temp <float>                temperature in [0,inf]; else 0.8");
        out.println("  --top-p <float>               top-p (nucleus) mass in (0,1]; else 0.95");
        out.println("  --top-k <int>                 top-k cutoff, 0 disables; else 40");
        out.println(
                "  --min-p <float>               min-p cutoff relative to the top token, in"
                        + " [0,1]; else 0.05");
        out.println(
                "  --seed, -s <long>             pins the sampling seed; default: a fresh random"
                        + " seed per request");
        out.println();
        out.println("Limits:");
        out.println(
                "  --context-capacity, -c <int>  allocated context positions (default: "
                        + PromptCache.Options.DEFAULT_CONTEXT_CAPACITY
                        + ", or the model's context length when smaller); 0 uses the model"
                        + " maximum; refused above the model's context length");
        out.println(
                "  --max-output-tokens, -n <int> how much it may produce in one turn; -1 (the"
                        + " default) = whatever the remaining context allows");
        out.println(
                "  --threads, -t <int>           compute threads, the one pool every kernel runs"
                        + " on (default: one per physical core; same as -Djinfer.threads)");
        out.println(
                "  --speculation-depth <int>     drafts per verify block for a model with a"
                        + " draft head (gemma4's MTP sidecar, attached with --with"
                        + " speculation=<file>); 0 disables, default 4");
        out.println(
                "  --cache <file>                persistent prompt cache (instruct/server) -"
                        + " serves matching prefixes, appends new prompts");
        out.println(
                "  --cache-ro <file>             like --cache but read-only - serves matching"
                        + " prefixes, never writes");
        out.println();
        out.println("Output:");
        out.println(
                "  --stream <boolean>            print tokens during generation; accepts"
                        + " true|false|on|off, default true");
        out.println(
                "  --echo <boolean>              print ALL tokens to stderr; accepts"
                        + " true|false|on|off, default false");
        out.println(
                "  --color <on|off|auto>         colorize thinking output in terminal (default:"
                        + " auto)");
        out.println();
        out.println("Server (with --server):");
        out.println("  --host <host>                 bind host, default 127.0.0.1");
        out.println("  --port <int>                  bind port, default " + DEFAULT_PORT);
        out.println(
                "  --api-key <token>             require a bearer token; mandatory off loopback");
        out.println(
                "  --cors-origin <origin>        allowed browser origin; repeatable, default *");
        out.println(
                "  --concurrency <int>           requests admitted at once (2x this many in"
                        + " flight, the rest get 503), default 16");
        out.println("  --queue-capacity <int>        waiting generations, default 4");
        out.println("  --max-body-mb <int>           request body limit, default 32");
        out.println("  --write-timeout <seconds>     stalled SSE write limit, default 30");
        out.println("  --request-timeout <seconds>   generation deadline, default 300; 0 disables");
        out.println("  --no-grammar                  reject grammar-constrained server requests");
        out.println();
        out.println("Interactive commands:");
        out.println("  /quit, /exit                  exit the chat");
        out.println("  /context                      show context token usage");
        out.println();
        out.println("Examples:");
        out.println("  java -jar jinfer.jar --model unsloth/gemma-4-E2B-it-GGUF:Q8_0" + " --chat");
        out.println(
                "  java -jar jinfer.jar --model unsloth/gemma-4-E2B-it-GGUF:Q8_0 --mmproj"
                        + " unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf --chat");
        out.println("  java -jar jinfer.jar pull ggml-org/stories15M_MOE:Q8_0");
        out.println(
                "  java -jar jinfer.jar --model"
                        + " LiquidAI/LFM2.5-350M-GGUF:Q8_0 --prompt \"Tell me a joke\"");
        out.println(
                "  java -jar jinfer.jar --model"
                        + " LiquidAI/LFM2.5-350M-GGUF:Q8_0 --chat"
                        + " --system-prompt \"You are a helpful assistant\"");
    }
}
