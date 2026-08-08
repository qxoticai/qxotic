package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.Server;
import com.qxotic.jinfer.server.ServerConfig;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.net.InetSocketAddress;
import java.nio.file.Path;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.TreeSet;

/**
 * The parsed command line: every flag, exactly as typed, with the four sampling knobs still
 * nullable because "the user said nothing" is information the model's own recommendations need.
 * {@link #parse} is the only way argv becomes one of these, and {@link #printUsage} documents the
 * same flags, so a new flag and its help line are added in the same file or noticed missing.
 *
 * <p>This is a CLI type, not a server one. It is wide because a command line is wide; what the
 * server needs is the narrow projection {@link #toServerConfig} builds, and nothing downstream of
 * that ever sees {@code --colors} or {@code --chat}. The two also validate different things: the
 * checks here exist to produce a good message next to a usage block, while {@link ServerConfig}
 * validates its own contract for any caller.
 *
 * @param maxOutputTokens tokens GENERATED per turn, -1 = as many as the context allows. The same
 *     meaning in every mode; in server mode it is the default for a request that omits {@code
 *     max_tokens}
 * @param contextCapacity the size of a session's state, and the ceiling on every one-shot. Refused
 *     above the model's own context length
 */
public record Options(
        Path modelPath,
        Map<String, Path> companions,
        Path tokenizerPath,
        Integer speculationDepth,
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
        int maxOutputTokens,
        int contextCapacity,
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
        companions = companions == null ? Map.of() : Map.copyOf(companions);
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
        require(contextCapacity >= 1, "Invalid argument: --context-capacity must be at least 1");
        require(
                maxOutputTokens >= -1,
                "Invalid argument: --max-output-tokens must be -1 (fill the context) or"
                        + " non-negative");
        // depth without the draft head would be a flag that does nothing (the --no-grammar rule)
        require(
                speculationDepth == null || companions.containsKey("speculation"),
                "Invalid argument: --speculation-depth needs --with speculation=<mtp file> - there"
                        + " is no draft head to run at any depth");
        require(
                speculationDepth == null || (1 <= speculationDepth && speculationDepth <= 8),
                "Invalid argument: --speculation-depth must be within [1, 8]");
        // the only thing --no-grammar does is refuse requests that ask for a grammar, and only
        // the HTTP API has requests. Accepting it elsewhere made it a flag that did nothing.
        require(
                !noGrammar || server,
                "Invalid argument: --no-grammar applies to --server (it refuses requests carrying"
                        + " grammar or response_format); there is nothing to refuse in chat or"
                        + " instruct mode");
    }

    /** The draft depth to speculate at: the flag, else 4 - the measured list/code sweet spot. */
    public int specDepth() {
        return speculationDepth == null ? 4 : speculationDepth;
    }

    /**
     * Refuses a capacity larger than the model was trained for - which needs the model, so it is
     * checked once, right after the load, rather than in the compact constructor with the flags
     * that stand on their own.
     */
    public void requireFitsModel(LoadedModel<?> model) {
        int trained = model.model().config().contextLength();
        require(
                contextCapacity <= trained,
                "Invalid argument: --context-capacity %d exceeds what %s was trained for (%d)",
                contextCapacity,
                modelPath.getFileName(),
                trained);
    }

    /**
     * The state size for a ONE-SHOT: exactly the work, never more. A single generation needs the
     * prompt plus its budget and not one position beyond, which is why instruct mode needs no size
     * flag of its own - and why the banner stopped carrying an arbitrary number.
     */
    public int oneShotCapacity(int promptLen) {
        return maxOutputTokens < 0
                ? contextCapacity
                : Math.max(16, Math.min(contextCapacity, promptLen + maxOutputTokens));
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
                new ServerConfig.Defaults(sampling(defaults), maxOutputTokens, think, rawPrompt),
                limits.withGrammar(!noGrammar).withSpeculationDepth(specDepth()),
                PromptCache.Options.DEFAULTS
                        .withContextCapacity(contextCapacity)
                        .withCatalog(promptCache, promptCacheReadOnly));
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
        long seconds = parseLong(optionName, value);
        require(seconds >= 0, "Invalid argument for %s: must be non-negative", optionName);
        return Duration.ofSeconds(seconds);
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

    public static boolean supportsAnsiColors(String colorMode) {
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

    /** llama.cpp's default too: big enough for real work, small enough to allocate blind. */
    static final int DEFAULT_CONTEXT_CAPACITY = 4096;

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
        Integer speculationDepth = null;
        // capability -> path or ref; resolved once parsing has succeeded
        Map<String, String> companionRefs = new LinkedHashMap<>();
        Long seed = null; // unset = a fresh random seed per request
        int maxOutputTokens = -1;
        int contextCapacity = DEFAULT_CONTEXT_CAPACITY;
        boolean interactive = false;
        boolean server = false;
        String host = "127.0.0.1";
        int port = 17341;
        boolean stream = true;
        boolean echo = false;
        boolean think = true;
        boolean thinkInline = false;
        String colorMode = "auto";
        boolean rawPrompt = false;
        boolean noGrammar = false;
        Path promptCache = null;
        boolean promptCacheReadOnly = false;
        ServerConfig.Limits limits = ServerConfig.Limits.DEFAULTS;

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
                        case "--temperature", "--temp" ->
                                temperature = parseFloat(optionName, nextArg);
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
                        // the server's ceilings. These were jinfer.server* system properties,
                        // which meant a flag and a -D could not both exist for one knob without
                        // one of them being a lie about precedence
                        case "--threads" ->
                                limits = limits.withThreads(parseInt(optionName, nextArg));
                        case "--queue-capacity" ->
                                limits = limits.withQueueCapacity(parseInt(optionName, nextArg));
                        case "--max-body-mb" ->
                                limits =
                                        limits.withMaxBodyBytes(
                                                (long) parseInt(optionName, nextArg) << 20);
                        case "--write-timeout" ->
                                limits = limits.withWriteTimeout(seconds(optionName, nextArg));
                        case "--request-timeout" ->
                                limits = limits.withRequestTimeout(seconds(optionName, nextArg));
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
                        case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                        case "--cache" -> promptCache = Path.of(nextArg);
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
            List<String> wanted = new java.util.ArrayList<>();
            if (modelRef != null) wanted.add(modelRef);
            if (tokenizerRef != null) wanted.add(tokenizerRef);
            for (String value : companionRefs.values()) {
                if (!"auto".equals(value)) wanted.add(value);
            }
            List<Path> resolved = ModelStore.resolveAll(wanted);
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
                speculationDepth,
                prompt,
                systemPrompt,
                interactive,
                server,
                host,
                port,
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
                color,
                rawPrompt,
                noGrammar,
                promptCache,
                promptCacheReadOnly,
                limits);
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
        out.println(
                "        java -jar jinfer.jar pull [--force] <ref>...  download models, print"
                        + " paths");
        out.println();
        out.println(
                "A remote model is a URL without the scheme: "
                        + " <host>/<owner>/<repo>[@rev][/path][:quant]");
        out.println(
                "  hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M     "
                        + " modelscope.cn/Qwen/Qwen3-0.6B-GGUF:Q8_0");
        out.println(
                "Hosts: hf.co, modelscope.cn. Quant defaults to Q4_K_M. Anything else is a local"
                        + " file path.");
        out.println(
                "Cache: $JINFER_MODELS, else the platform cache dir. HF_TOKEN for gated repos,"
                        + " JINFER_OFFLINE=1 to never fetch.");
        out.println();
        out.println("Options:");
        out.println(
                "  --model, -m <path|ref>        required, a .gguf file or a remote model -"
                        + " downloaded on first use");
        out.println("  --mmproj <path|ref>           shorthand for --with media=<...>");
        out.println(
                "  --with <role>=<path|ref>      attach a file by role. model= (same as -m) and"
                        + " tokenizer= (use another GGUF's tokenizer; id-space checked) are"
                        + " reserved; any other role is a COMPANION capability of the"
                        + " architecture, e.g. gemma4: media (vision/audio encoders)");
        out.println("  --interactive, --chat, -i     run in chat mode");
        out.println("  --instruct                    run in instruct (once) mode, default mode");
        out.println("  --server                      run an OpenAI-compatible HTTP server");
        out.println("  --host <host>                 server bind host, default 127.0.0.1");
        out.println("  --port <int>                  server bind port, default 17341");
        out.println("  --threads <int>               server handler threads, default 16");
        out.println(
                "  --queue-capacity <int>        generation requests that may WAIT, default 4 (0 ="
                        + " reject unless idle)");
        out.println("  --max-body-mb <int>           request body limit, default 32");
        out.println("  --write-timeout <seconds>     streaming write stall limit, default 30");
        out.println("  --request-timeout <seconds>   generation deadline, default 300 (0 = none)");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
        out.println(
                "  --temperature, --temp <float> temperature in [0,inf]; default: the model's"
                        + " recommended value, else 0.8");
        out.println(
                "  --top-p <float>               top-p (nucleus) mass in [0,1]; default: the"
                        + " model's recommended value, else 0.95");
        out.println(
                "  --top-k <int>                 top-k cutoff, 0 disables; default: the model's"
                        + " recommended value, else 40");
        out.println(
                "  --min-p <float>               min-p cutoff relative to the top token, in"
                        + " [0,1]; default: the model's recommended value, else 0.05");
        out.println(
                "  --seed, -s <long>             pins the sampling seed; default: a fresh random"
                        + " seed per request");
        out.println(
                "  --context-capacity, -c <int>  how much the model can remember, in tokens;"
                        + " default "
                        + DEFAULT_CONTEXT_CAPACITY
                        + ", refused above the model's own context length");
        out.println(
                "  --speculation-depth <int>     draft tokens per verify with --with speculation="
                        + " (gemma4 + its mtp sidecar); default 4. ~1.7x on code/lists with Q8_0;"
                        + " can slow prose, and k-quants (Q4_K_M) verify slowly today");
        out.println(
                "  --max-output-tokens, -n <int> how much it may produce in one turn; -1 (the"
                        + " default) = whatever the remaining context allows. A one-shot --prompt"
                        + " allocates only what it needs: prompt + this");
        out.println(
                "  --stream <boolean>            print tokens during generation; accepts"
                        + " true|false|on|off, default true");
        out.println(
                "  --echo <boolean>              print ALL tokens to stderr; accepts"
                        + " true|false|on|off, default false");
        out.println(
                "  --color <on|off|auto>         colorize thinking output in terminal (default:"
                        + " auto)");
        out.println(
                "  --think <off|on|inline>       on: show thinking (default), off: hide thinking"
                        + " from output (model still generates it), inline: thoughts to stdout");
        out.println(
                "  --raw-prompt                  bypass chat template and tokenize --prompt"
                        + " directly");
        out.println(
                "  --cache <file>                persistent prompt cache (instruct + server) -"
                        + " serves matching prefixes, appends new prompts");
        out.println(
                "  --cache-ro <file>             like --cache but read-only - serves matching"
                        + " prefixes, never writes");
        out.println();
        out.println("Interactive commands:");
        out.println("  /quit, /exit                  exit the chat");
        out.println("  /context                      show context token usage");
        out.println();
        out.println("Examples:");
        out.println("  java -jar jinfer.jar --model hf.co/unsloth/gemma-4-E2B-it-GGUF --chat");
        out.println(
                "  java -jar jinfer.jar --model hf.co/unsloth/gemma-4-E2B-it-GGUF --mmproj"
                        + " hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf --server");
        out.println("  java -jar jinfer.jar pull hf.co/ggml-org/stories15M_MOE:Q8_0");
        out.println("  java -jar jinfer.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat");
        out.println(
                "  java -jar jinfer.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt \"Tell me a"
                        + " joke\"");
        out.println(
                "  java -jar jinfer.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat"
                        + " --system-prompt \"You are a helpful assistant\"");
        out.println(
                "  java -jar jinfer.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --server --port"
                        + " 17341");
    }
}
