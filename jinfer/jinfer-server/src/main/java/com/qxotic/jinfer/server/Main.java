// jinfer: LLM inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat, --instruct, and --server
//
// Build/run: `mvn package` then `java -jar target/jinfer.jar --help` (see the Makefile for the
// exact runtime flags and native-image targets).
package com.qxotic.jinfer.server;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.*;
import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.Conversation;
import com.qxotic.jinfer.chat.JinjaChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.kernels.*;
import com.qxotic.jinfer.llm.*;
import com.qxotic.toknroll.IntSequence;
import com.qxotic.toknroll.Tokenizer;
import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.IntConsumer;

public class Main {

    private static final String ANSI_GREY = "\033[90m";
    private static final String ANSI_CYAN = "\033[36m";
    private static final String ANSI_RESET = "\033[0m";

    private static void onThinkingStart(PrintStream thoughtOut, boolean ansi) {
        if (ansi) {
            thoughtOut.print(ANSI_GREY);
        }
        thoughtOut.println("[Start thinking]");
    }

    private static void onThinkingEnd(PrintStream thoughtOut, boolean ansi, boolean emitted) {
        if (emitted) {
            thoughtOut.println();
        }
        thoughtOut.println("[End thinking]");
        if (ansi) {
            thoughtOut.print(ANSI_RESET);
        }
        thoughtOut.println();
    }

    private static IntConsumer streamingPrinter(Tokenizer tokenizer, LLMOptions options) {
        if (!options.stream()) {
            return token -> {};
        }

        var open = SpecialTokens.find(tokenizer, "<think>");
        var close = SpecialTokens.find(tokenizer, "</think>");
        if (open.isEmpty() || close.isEmpty()) {
            return token -> { // no think markers in the vocabulary: plain content streaming
                if (!SpecialTokens.isSpecial(tokenizer, token)) {
                    byte[] bytes = tokenizer.decodeBytes(new int[] {token});
                    System.out.write(bytes, 0, bytes.length);
                }
            };
        }

        int thinkOpen = open.getAsInt();
        int thinkClose = close.getAsInt();
        boolean thinkEnabled = options.think();
        PrintStream thoughtOut = options.thinkInline() ? System.out : System.err;
        boolean ansi = options.colors();
        boolean[] inThink = {false};
        boolean[] emitted = {false};
        return token -> {
            if (token == thinkOpen) {
                if (thinkEnabled) {
                    onThinkingStart(thoughtOut, ansi);
                }
                inThink[0] = true;
                emitted[0] = false;
                return;
            }
            if (token == thinkClose) {
                if (thinkEnabled) {
                    onThinkingEnd(thoughtOut, ansi, emitted[0]);
                }
                inThink[0] = false;
                emitted[0] = false;
                return;
            }
            if (!SpecialTokens.isSpecial(tokenizer, token)) {
                byte[] bytes = tokenizer.decodeBytes(new int[] {token});
                if (inThink[0]) {
                    if (thinkEnabled) {
                        thoughtOut.write(bytes, 0, bytes.length);
                        emitted[0] = true;
                    }
                } else {
                    System.out.write(bytes, 0, bytes.length);
                }
            }
        };
    }

    /**
     * A CLI generation outcome: the raw result, the display text the parser assembled, and the
     * parser's structured reply message (verbatim ids - what a codec chat loop appends).
     */
    private record CliReply(Generator.GenerationResult result, String text, Message message) {}

    /**
     * One generation pass plus CLI presentation: prompt echo, token streaming through the printer,
     * and the stderr timing summary line. --max-tokens is a TOTAL context cap in the CLI, so it is
     * converted to the generator's completion budget here. The display text is assembled by a plain
     * span parser (the CLI offers no tools): think content is bracketed inline when thinking is
     * shown, dropped otherwise.
     */
    private static <S extends RuntimeState> CliReply generateCli(
            LoadedModel<S> model,
            S state,
            IntSequence promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            LLMOptions options) {
        return generateCli(model, state, promptTokens, stopTokens, sampler, options, null);
    }

    private static <S extends RuntimeState> CliReply generateCli(
            LoadedModel<S> model,
            S state,
            IntSequence promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            LLMOptions options,
            java.util.function.IntConsumer afterIngest) {
        Tokenizer tokenizer = model.tokenizer();
        if (options.echo()) {
            echoPrompt(tokenizer, promptTokens);
        }
        IntConsumer printer = streamingPrinter(tokenizer, options);
        IntConsumer onToken =
                !options.echo()
                        ? printer
                        : token -> {
                            System.err.print(
                                    replaceControlCharacters(tokenizer.decode(new int[] {token})));
                            printer.accept(token);
                        };
        int startPosition = state.position();
        int budget =
                options.maxTokens() < 0
                        ? -1
                        : options.maxTokens() - (startPosition + promptTokens.length());
        int totalPrompt = promptTokens.length();
        ReplyParser parser = ReplyParser.spans(tokenizer);
        StringBuilder text = new StringBuilder();
        InlineThink inlineThink = new InlineThink();
        java.util.function.BiConsumer<String, Boolean> collect =
                (fragment, reasoning) -> {
                    if (!reasoning) {
                        text.append(
                                options.think() ? inlineThink.project(fragment, false) : fragment);
                    } else if (options.think()) {
                        // thinking shown: bracket it inline in the display text (the old
                        // visible-tokens rendering kept think spans when --think is on)
                        text.append(inlineThink.project(fragment, true));
                    }
                };
        Generator.GenerationResult result =
                Generator.generate(
                        model.model(),
                        state,
                        promptTokens.isEmpty()
                                ? List.of()
                                : List.of(Batch.prefill(promptTokens.toArray())),
                        sampler,
                        budget,
                        0 /* CLI: no deadline */,
                        stopTokens,
                        token -> {
                            onToken.accept(token);
                            String fragment = parser.feed(token);
                            if (!fragment.isEmpty()) collect.accept(fragment, parser.reasoning());
                            return true;
                        },
                        afterIngest);
        Message message = parser.finish();
        int generated = result.tokens().length() + (result.stopToken() >= 0 ? 1 : 0);
        String timingPrefix = options.colors() ? ANSI_CYAN : "";
        String timingSuffix = options.colors() ? ANSI_RESET : "";
        System.err.printf(
                "%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + totalPrompt + generated,
                model.model().config().contextLength(),
                totalPrompt / (result.promptNanos() / 1e9),
                totalPrompt,
                generated / (result.predictedNanos() / 1e9),
                generated,
                timingSuffix);
        return new CliReply(result, text.toString(), message);
    }

    /** {@code --echo}: the prompt tokens to stderr, control characters escaped. */
    private static void echoPrompt(Tokenizer tokenizer, IntSequence promptTokens) {
        promptTokens.forEachInt(
                token ->
                        System.err.print(
                                replaceControlCharacters(tokenizer.decode(new int[] {token}))));
    }

    /** Escape control characters (except newline) so token echo cannot distort the terminal. */
    private static String replaceControlCharacters(String str) {
        StringBuilder chars = new StringBuilder();
        str.codePoints()
                .forEach(
                        cp -> {
                            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
                            } else {
                                chars.appendCodePoint(cp);
                            }
                        });
        return chars.toString();
    }

    static final int DEFAULT_MAX_TOKENS = 1024;

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
                "  --with <capability>=<file>    attach a COMPANION that gives the model a"
                        + " capability: a path or a ref, named explicitly. gemma4: media"
                        + " (vision/audio encoders)");
        out.println("  --interactive, --chat, -i     run in chat mode");
        out.println("  --instruct                    run in instruct (once) mode, default mode");
        out.println("  --server                      run an OpenAI-compatible HTTP server");
        out.println("  --host <host>                 server bind host, default 127.0.0.1");
        out.println("  --port <int>                  server bind port, default 17325");
        out.println("  --prompt, -p <string>         input prompt");
        out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
        out.println(
                "  --temperature, -temp <float>  temperature in [0,inf]; default: the model's"
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
                "  --seed <long>                 pins the sampling seed; default: a fresh random"
                        + " seed per request");
        out.println(
                "  --max-tokens, -n <int>        number of steps to run for < 0 = limited by"
                        + " context length, default "
                        + DEFAULT_MAX_TOKENS);
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
                        + " 17325");
    }

    static LLMOptions parseOptions(String[] args) {
        String prompt = null;
        String systemPrompt = null;
        Float temperature = null; // unset = the model's recommended value, else 0.8
        Float topp = null; // unset = the model's recommended value, else 0.95
        Integer topk = null; // unset = the model's recommended value, else 40
        Float minp = null; // unset = the model's recommended value, else 0.05
        // paths or hub refs; resolved (and downloaded, if needed) once parsing has succeeded
        String modelRef = null;
        // capability -> path or ref; resolved once parsing has succeeded
        java.util.Map<String, String> companionRefs = new java.util.LinkedHashMap<>();
        Long seed = null; // unset = a fresh random seed per request
        int maxTokens = DEFAULT_MAX_TOKENS;
        boolean interactive = false;
        boolean server = false;
        String host = "127.0.0.1";
        int port = 17325;
        boolean stream = true;
        boolean echo = false;
        boolean think = true;
        boolean thinkInline = false;
        String colorMode = "auto";
        boolean rawPrompt = false;
        boolean noGrammar = false;
        Path promptCache = null;
        boolean promptCacheReadOnly = false;

        for (int i = 0; i < args.length; i++) {
            String optionName = args[i];
            LLMOptions.require(optionName.startsWith("-"), "Invalid option %s", optionName);
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
                        LLMOptions.require(
                                i + 1 < args.length, "Missing argument for option %s", optionName);
                        nextArg = args[i + 1];
                        i += 1;
                    }
                    switch (optionName) {
                        case "--prompt", "-p" -> prompt = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                        case "--top-p" -> topp = Float.parseFloat(nextArg);
                        case "--top-k" -> topk = Integer.parseInt(nextArg);
                        case "--min-p" -> minp = Float.parseFloat(nextArg);
                        case "--model", "-m" -> modelRef = nextArg;
                        case "--mmproj" -> companionRefs.put("media", nextArg);
                        case "--with" -> {
                            int eq = nextArg.indexOf('=');
                            LLMOptions.require(
                                    eq > 0 && eq < nextArg.length() - 1,
                                    "--with takes <capability>=<path|ref>, got %s",
                                    nextArg);
                            companionRefs.put(nextArg.substring(0, eq), nextArg.substring(eq + 1));
                        }
                        case "--host" -> host = nextArg;
                        case "--port" -> port = Integer.parseInt(nextArg);
                        case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                        case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                        case "--stream" ->
                                stream = LLMOptions.parseBooleanOption(optionName, nextArg);
                        case "--echo" -> echo = LLMOptions.parseBooleanOption(optionName, nextArg);
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
                                        LLMOptions.require(
                                                false,
                                                "Invalid argument for %s: expected off|on|inline"
                                                        + " (or false|true|stdout), got %s",
                                                optionName,
                                                nextArg);
                            }
                        }
                        default -> LLMOptions.require(false, "Unknown option: %s", optionName);
                    }
                }
            }
        }
        LLMOptions.require(
                List.of("on", "off", "auto").contains(colorMode),
                "Invalid argument: --color must be one of on|off|auto");
        boolean color = LLMOptions.supportsAnsiColors(colorMode);
        // AFTER the loop, so --help and a bad flag never trigger a download first
        Path modelPath;
        java.util.Map<String, Path> companions = new java.util.LinkedHashMap<>();
        try {
            modelPath = modelRef == null ? null : ModelStore.resolve(modelRef);
            if (!companionRefs.isEmpty()) {
                // ONE header read for all of them, and the capability is checked before any file
                // is fetched: a wrong knob should fail on the knob, not on a missing download
                java.util.Map<String, String> offered;
                try {
                    offered = Models.companionFiles(modelPath);
                } catch (IOException e) {
                    throw new java.io.UncheckedIOException(e);
                }
                for (var wanted : companionRefs.entrySet()) {
                    companions.put(
                            wanted.getKey(),
                            resolveCompanion(offered, wanted.getKey(), wanted.getValue()));
                }
            }
        } catch (RuntimeException e) {
            // a ref that cannot be resolved carries its own remedy (gated repo, unknown quant,
            // no such repository); the usage block would only bury it
            throw new ResolveFailure(e);
        }
        return new LLMOptions(
                modelPath,
                java.util.Map.copyOf(companions),
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
                maxTokens,
                stream,
                echo,
                think,
                thinkInline,
                color,
                rawPrompt,
                noGrammar,
                promptCache,
                promptCacheReadOnly);
    }

    /**
     * Force UTF-8 on the console so multilingual model output/input isn't garbled by a legacy code
     * page (Windows defaults stdout/stdin to one; Linux/macOS are already UTF-8, so this is a no-op
     * re-wrap). Raw-byte writes — the streamed token bytes — pass through unchanged; only String
     * prints are affected. Buffered + auto-flush, matching the default {@code System.out}.
     */
    private static void forceUtf8Console() {
        System.setOut(utf8Stream(FileDescriptor.out));
        System.setErr(utf8Stream(FileDescriptor.err));
    }

    /**
     * One line per log record, since this is a console tool: java.util.logging's default is a
     * two-line record led by a date, the logging class and the method, which buries the message
     * that matters. Only a default - an explicit {@code -D} wins, and an embedder running its own
     * backend never reaches java.util.logging at all.
     */
    private static void oneLineLogs() {
        String format = "java.util.logging.SimpleFormatter.format";
        if (System.getProperty(format) == null) {
            System.setProperty(format, "%1$tT %4$-7s %5$s%6$s%n");
        }
    }

    private static PrintStream utf8Stream(FileDescriptor fd) {
        return new PrintStream(
                new BufferedOutputStream(new FileOutputStream(fd), 8192),
                true,
                StandardCharsets.UTF_8);
    }

    /**
     * One companion, from whatever the flag said. {@code auto} asks the model's ARCHITECTURE which
     * filename carries the capability, then asks the hub to find that file in the model's own
     * repository - so a user names a capability and never a filename.
     */
    /**
     * One companion, named EXPLICITLY. There is no "find it for me": a companion changes what the
     * model produces - a projector is in the media cache key - so which file it is belongs in the
     * command line where it can be read, reproduced and pinned.
     */
    private static Path resolveCompanion(
            java.util.Map<String, String> offered, String capability, String value) {
        String fileName = offered.get(capability);
        LLMOptions.require(
                fileName != null,
                "this model has no '%s' capability. It offers: %s",
                capability,
                offered.isEmpty() ? "none" : new java.util.TreeSet<>(offered.keySet()));
        LLMOptions.require(
                !"auto".equals(value),
                "name the '%s' file rather than 'auto' - it is usually called %s*, and browsing"
                        + " the model's repository shows which precisions it ships",
                capability,
                fileName);
        return ModelStore.resolve(value);
    }

    /** A model ref that could not be resolved: the cause already says what to do about it. */
    private static final class ResolveFailure extends RuntimeException {
        ResolveFailure(RuntimeException cause) {
            super(cause.getMessage(), cause); // the cause's own words, not its class name
        }
    }

    /**
     * Downloads each ref and prints where it landed, one path per line, so the output pipes. The
     * only thing {@code -m <ref>} does not already do implicitly - it exists to warm a CI image or
     * a laptop before a flight.
     */
    private static void pull(String[] args) {
        boolean force = false;
        List<String> refs = new ArrayList<>();
        for (String arg : args) {
            if (arg.equals("--force") || arg.equals("-f")) {
                force = true;
            } else {
                refs.add(arg);
            }
        }
        if (refs.isEmpty()) {
            System.err.println("ERROR pull needs at least one model ref, e.g.");
            System.err.println("  jinfer pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M");
            System.err.println("  jinfer pull --force <ref>    re-download even if cached");
            System.exit(2);
            return;
        }
        for (String ref : refs) {
            try {
                if (force) {
                    ModelStore.evict(ref);
                }
                System.out.println(ModelStore.resolve(ref));
            } catch (RuntimeException e) {
                System.err.println("ERROR " + message(e));
                System.exit(1);
                return;
            }
        }
    }

    /** The remedy a resolution failure carries, unwrapping the plumbing around it. */
    private static String message(Throwable failure) {
        Throwable root = failure;
        while (root.getMessage() == null && root.getCause() != null) {
            root = root.getCause();
        }
        return root.getMessage();
    }

    /**
     * What the local cache holds, as refs with their sizes. Every line pastes straight back into
     * {@code --model} or {@code --with}, which is the point: the cache path IS the ref.
     */
    private static void list() {
        java.util.List<String> refs = ModelStore.cached();
        if (refs.isEmpty()) {
            System.out.println("no models cached in " + ModelStore.root());
            return;
        }
        int width = refs.stream().mapToInt(String::length).max().orElse(0);
        long total = 0;
        for (String ref : refs) {
            long size = sizeOf(ModelStore.root().resolve(ref)); // resolve() ignores an absolute ref
            total += size;
            System.out.printf("%-" + width + "s  %10s%n", ref, humanBytes(size));
        }
        System.out.printf("%-" + width + "s  %10s%n", "total", humanBytes(total));
    }

    private static long sizeOf(Path file) {
        try {
            return Files.size(file);
        } catch (IOException unreadable) {
            return 0;
        }
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        String[] units = {"KB", "MB", "GB", "TB"};
        double value = bytes;
        int unit = -1;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit++;
        }
        return String.format(Locale.ROOT, value >= 100 ? "%.0f %s" : "%.1f %s", value, units[unit]);
    }

    public static void main(String[] args) throws IOException {
        forceUtf8Console();
        oneLineLogs();
        if (args.length > 0 && !args[0].startsWith("-")) {
            if (args[0].equals("list")) {
                LLMOptions.require(args.length == 1, "list takes no arguments");
                list();
                return;
            }
            if (!args[0].equals("pull")) {
                System.err.println(
                        "ERROR unknown command: " + args[0] + " (the only one is: pull)");
                System.err.println();
                printUsage(System.err);
                System.exit(2);
                return;
            }
            pull(java.util.Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        LLMOptions options;
        try {
            options = parseOptions(args);
        } catch (ResolveFailure e) {
            System.err.println("ERROR " + message(e));
            System.exit(1);
            return;
        } catch (IllegalArgumentException e) {
            System.err.println("ERROR " + e.getMessage());
            System.err.println();
            printUsage(System.err);
            System.exit(1);
            return;
        }
        LoadedModel<?> model;
        try {
            if (!options.companions().isEmpty()) {
                // a model with companions is never AOT-preloaded: load the set fresh
                model =
                        Models.load(
                                options.modelPath(),
                                options.maxTokens(),
                                java.lang.foreign.Arena.global(),
                                options.companions());
            } else {
                model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
                if (model == null) {
                    model =
                            Models.load(
                                    options.modelPath(),
                                    options.maxTokens(),
                                    java.lang.foreign.Arena.global());
                }
            }
        } catch (IllegalArgumentException
                | IllegalStateException
                | UnsupportedOperationException
                | java.io.UncheckedIOException
                | java.nio.file.NoSuchFileException e) {
            // load errors carry their remedy in the message (wrong mmproj, unknown architecture,
            // split GGUF, bad pre-tokenizer flag, ...) - print it, don't bury it in a stack
            // trace; anything else is a bug and still traces
            System.err.println("ERROR " + e.getMessage());
            System.exit(1);
            return;
        }
        LLMOptions resolved = options.withResolvedSampling(model.samplingDefaults());
        if (options.server()) {
            serve(model, options, resolved);
            return;
        }
        options = resolved;
        Sampler sampler =
                Sampler.select(
                        model.model().config().vocabularySize(),
                        options.temperature(),
                        options.topk(),
                        options.topp(),
                        options.minp(),
                        options.seed() != null
                                ? options.seed()
                                : java.util.concurrent.ThreadLocalRandom.current().nextLong());
        if (!options.think()) {
            sampler = Thinking.banMarkers(sampler, model.tokenizer());
        }
        runGeneric(model, sampler, options);
    }

    /**
     * Serves over HTTP, and owns the presentation {@link Server} deliberately has none of: what is
     * being served, where each sampling value came from, and - last, because it is the one line a
     * reader acts on - the endpoint, whose port only exists once {@code start} has bound it.
     *
     * <p>The classpath's full architecture list used to go in the first line, which answered a
     * question nobody asks at startup; it belongs in the "no provider for architecture" error,
     * where it already is.
     */
    private static void serve(LoadedModel<?> model, LLMOptions options, LLMOptions resolved)
            throws IOException {
        System.out.printf(
                "model       %s  (%s, ctx %d)%n",
                options.modelPath().getFileName(),
                model.model().getClass().getSimpleName(),
                model.model().config().contextLength());
        options.companions()
                .forEach(
                        (capability, file) ->
                                System.out.printf(
                                        "companion   %s = %s%n", capability, file.getFileName()));
        var defaults = model.samplingDefaults();
        System.out.printf(
                "sampling    temperature %s, top-k %s, top-p %s, min-p %s; requests override%n",
                describe(
                        resolved.temperature(),
                        options.temperature() != null,
                        defaults.temperature() != null),
                describe(resolved.topk(), options.topk() != null, defaults.topK() != null),
                describe(resolved.topp(), options.topp() != null, defaults.topP() != null),
                describe(resolved.minp(), options.minp() != null, defaults.minP() != null));
        Server.Running running = Server.start(model, resolved);
        // the CLI never closes the handle; ^C must still free the engine deterministically
        Runtime.getRuntime().addShutdownHook(new Thread(running::close));
        System.out.printf(
                "listening   http://%s:%d  (OpenAI-compatible)%n",
                options.host(), running.address().getPort());
    }

    /**
     * A resolved sampling value with its provenance: the user's flag, the model (GGUF metadata or
     * its port author's recommendation), or jinfer's baseline - so a surprising default explains
     * itself instead of being blamed on the server.
     */
    private static String describe(Number value, boolean userSet, boolean modelRecommended) {
        String source =
                userSet ? "set by user" : modelRecommended ? "model default" : "jinfer default";
        String shown = value instanceof Float f ? String.valueOf(Server.trim(f)) : value.toString();
        return shown + " (" + source + ")";
    }

    /**
     * CLI driver: a one-shot {@code --prompt} or an interactive {@code --chat} loop. Models
     * exposing a {@link ChatTemplate} run through the model's own codec framing - native codecs
     * incrementally via whole-conversation re-encode + longest-common-prefix reuse (the verbatim
     * splice keeps generated turns in the common prefix); everything else falls back to the
     * whole-render Jinja path with a fresh state per turn.
     */
    static <S extends RuntimeState> void runGeneric(
            LoadedModel<S> model, Sampler sampler, LLMOptions options) throws IOException {
        ChatTemplate template = options.rawPrompt() ? null : model.template().orElse(null);
        if (!options.interactive()) {
            runInstruct(model, template, sampler, options);
        } else if (template != null) {
            runChatCodec(model, template, sampler, options);
        } else {
            runChatWholeRender(model, sampler, options);
        }
    }

    private static <S extends RuntimeState> void runInstruct(
            LoadedModel<S> model, ChatTemplate template, Sampler sampler, LLMOptions options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        IntSequence promptTokens;
        if (options.rawPrompt()) {
            promptTokens = SpecialTokens.encode(model.tokenizer(), options.prompt());
        } else if (template != null) {
            List<Message> turns = new ArrayList<>();
            if (options.systemPrompt() != null) {
                turns.add(Message.system(options.systemPrompt()));
            }
            turns.add(Message.user(options.prompt()));
            List<Batch> batches =
                    template.encode(new Conversation(turns, List.of(), options.think(), ""));
            promptTokens = IntSequence.wrap(Batch.tokenIds(batches));
        } else {
            List<Object> messages = new ArrayList<>();
            if (options.systemPrompt() != null) {
                messages.add(Map.of("role", "system", "content", options.systemPrompt()));
            }
            messages.add(Map.of("role", "user", "content", options.prompt()));
            promptTokens =
                    new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource())
                            .render(messages, null, true, options.think(), null);
        }
        // --cache / --cache-ro: the prompt cache as a file, through the one facade - which owns
        // the whole policy (codec-less models warn and serve without it, coarse codecs restore
        // read-only, a missing read-only file degrades). Read-write pins the prompt via define()
        // (fine codecs: chunk blocks + a split-last single; coarse: one residue block) and
        // appends the new blocks BEFORE generating - the artifact is the point of --cache, and a
        // generation failure must not lose it. serve() then restores the longest cached prefix
        // (one short, by the law) and generates on top.
        if (options.promptCache() != null && promptTokens.length() >= 2) {
            List<Batch> prompt = List.of(Batch.prefill(promptTokens.toArray()));
            int total = promptTokens.length();
            if (options.echo()) {
                // the whole prompt: serve() ingests it internally, so generateCli sees none of it
                echoPrompt(model.tokenizer(), promptTokens);
            }
            try (PromptCache<S> cache =
                    PromptCache.of(
                            model.model(),
                            model.seed(),
                            new PromptCache.Options(
                                    0,
                                    Long.MAX_VALUE,
                                    options.promptCache(),
                                    options.promptCacheReadOnly()))) {
                if (!options.promptCacheReadOnly() && cache.blockCaching()) {
                    int before = cache.sample().blocks();
                    cache.define(prompt);
                    cache.save();
                    int added = cache.sample().blocks() - before;
                    if (added > 0) {
                        System.err.printf(
                                "cache: %d blocks added, catalog appended (%s)%n",
                                added, options.promptCache());
                    }
                }
                long t0 = System.nanoTime();
                CliReply reply =
                        cache.serve(
                                prompt,
                                (state, serving) -> {
                                    System.err.printf(
                                            "cache: %d/%d positions restored, prompt ready in"
                                                    + " %.1f ms%n",
                                            serving.restored(),
                                            total,
                                            (System.nanoTime() - t0) / 1e6);
                                    return generateCli(
                                            model,
                                            state,
                                            IntSequence.empty(),
                                            stops,
                                            sampler,
                                            options,
                                            serving::tail);
                                });
                if (!options.stream()) {
                    System.out.println(reply.text());
                }
            }
            return;
        }

        S state = Generator.stateFor(model.model(), promptTokens.length());
        CliReply reply = generateCli(model, state, promptTokens, stops, sampler, options);
        if (!options.stream()) {
            System.out.println(reply.text());
        }
    }

    /**
     * Interactive chat on a NATIVE codec: ONE running {@link Conversation}, re-encoded whole each
     * turn; the longest common prefix with the token stream the KV already holds is skipped and
     * only the suffix is ingested. Replies are appended as the parser's structured message
     * (verbatim ids), so the codec's verbatim splice keeps every generated turn inside the common
     * prefix - the append-only happy path ingests exactly closeTurn + the user turn + the scaffold,
     * like the per-turn flow. Any divergence rebuilds the state from scratch (correctness first;
     * the splice makes it rare).
     */
    private static <S extends RuntimeState> void runChatCodec(
            LoadedModel<S> model, ChatTemplate template, Sampler sampler, LLMOptions options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        int contextLength = model.model().config().contextLength();
        S state = model.model().newState(contextLength, RuntimeFlags.BATCH_CAPACITY);
        List<Message> opening = new ArrayList<>();
        if (options.systemPrompt() != null) {
            opening.add(Message.system(options.systemPrompt()));
        }
        Conversation conversation = new Conversation(opening, List.of(), options.think(), "");
        IntSequence ingested = IntSequence.empty(); // the token stream the KV holds
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                if ("/context".equals(userText)) {
                    System.out.printf("context: %d/%d tokens%n", state.position(), contextLength);
                    continue;
                }
                conversation = conversation.append(Message.user(userText));
                IntSequence prompt =
                        IntSequence.wrap(Batch.tokenIds(template.encode(conversation)));
                int lcp = commonPrefix(ingested, prompt);
                IntSequence delta;
                if (lcp < ingested.length()) {
                    state = model.model().newState(contextLength, RuntimeFlags.BATCH_CAPACITY);
                    delta = prompt;
                } else {
                    delta = prompt.subSequence(lcp, prompt.length());
                }
                CliReply reply = generateCli(model, state, delta, stops, sampler, options);
                if (!options.stream()) {
                    System.out.println(reply.text());
                }
                conversation = conversation.append(reply.message());
                // The KV holds the prompt plus every INGESTED reply token: all of them when a
                // stop token ended the turn, all but the last otherwise (the decode loop never
                // ingests the final sampled token).
                IntSequence generated = reply.result().tokens();
                if (reply.result().stopToken() < 0 && !generated.isEmpty()) {
                    generated = generated.subSequence(0, generated.length() - 1);
                }
                ingested = prompt.concat(generated);
            }
        }
    }

    private static int commonPrefix(IntSequence a, IntSequence b) {
        int n = Math.min(a.length(), b.length());
        int i = 0;
        while (i < n && a.intAt(i) == b.intAt(i)) i++;
        return i;
    }

    /**
     * Whole-render fallback for models without a TurnTemplate: re-encode the full conversation
     * through the Jinja template each turn, fresh state.
     */
    private static <S extends RuntimeState> void runChatWholeRender(
            LoadedModel<S> model, Sampler sampler, LLMOptions options) throws IOException {
        Set<Integer> stops = model.stopTokens();
        JinjaChatTemplate jinja =
                new JinjaChatTemplate(model.tokenizer(), model.chatTemplateSource());
        List<Object> history = new ArrayList<>();
        if (options.systemPrompt() != null) {
            history.add(Map.of("role", "system", "content", options.systemPrompt()));
        }
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                history.add(Map.of("role", "user", "content", userText));
                IntSequence promptTokens = jinja.render(history, null, true, options.think(), null);
                CliReply reply =
                        generateCli(
                                model,
                                Generator.stateFor(model.model(), promptTokens.length()),
                                promptTokens,
                                stops,
                                sampler,
                                options);
                if (!options.stream()) {
                    System.out.println(reply.text());
                }
                history.add(Map.of("role", "assistant", "content", reply.text()));
            }
        }
    }
}

final class AOT {
    // The preloaded model's parsed GGUF (metadata + tensor descriptors), baked at class-init. In a
    // native image (AOT class initialized-at-build-time) this skips re-reading and re-parsing the
    // header at startup; the tensor data is still mmap'd at runtime. Arch-agnostic: any new-API
    // port
    // loads from it via Models.load(fileChannel, gguf, ctx).
    //
    // Tradeoff vs the old per-model AOT: that one baked the fully materialized tokenizer + config
    // and
    // only mmap'd weights at runtime. This generic version bakes the parsed GGUF and rebuilds the
    // tokenizer at runtime (Models.load re-materializes it), so the win is skipping the header
    // parse,
    // not the tokenizer build. A fuller bake would need a per-port "attach weights to a preloaded
    // config-only model" method across all ports; deferred.
    record PartialModel(String modelFileName, GGUF gguf) {}

    private static final PartialModel PRELOADED_GGUF =
            preLoadGGUF(System.getProperty("jinfer.PreloadGGUF"));

    private static PartialModel preLoadGGUF(String modelPath) {
        if (modelPath == null || modelPath.isEmpty()) {
            return null;
        }
        try {
            Path path = Path.of(modelPath);
            if (!Files.exists(path) || !Files.isRegularFile(path)) {
                throw new IllegalArgumentException("Cannot pre-load model: " + path);
            }
            try (FileChannel fileChannel = FileChannel.open(path, StandardOpenOption.READ)) {
                GGUF gguf = ModelLoader.readGguf(fileChannel, path.toString());
                return new PartialModel(path.getFileName().toString(), gguf);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * The preloaded model when {@code modelPath} matches the baked one, else null (the caller falls
     * back to {@link Models#load(Path, int)}). Reuses the baked GGUF, so only the tensor data is
     * read.
     */
    static LoadedModel<?> tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        PartialModel preLoaded = PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        if (!Objects.equals(modelPath.getFileName().toString(), preLoaded.modelFileName())) {
            return null;
        }
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return Models.load(
                    fileChannel, preLoaded.gguf(), contextLength, java.lang.foreign.Arena.global());
        }
    }
}
