// LFM2.5 inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat, --instruct, and --server
//
// Build/run: `mvn package` then `java -jar target/lfm25.jar --help` (see the Makefile for the
// exact runtime flags and native-image targets).
package com.llama4j;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.format.gguf.TensorEntry;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintStream;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.HexFormat;
import java.util.function.IntConsumer;

public class LFM25 {

    private static final String ANSI_GREY  = "\033[90m";
    private static final String ANSI_CYAN  = "\033[36m";
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

    static boolean supportsAnsiColors(String colorMode) {
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

    private static IntConsumer streamingPrinter(LFMTokenizer tokenizer, Options options) {
        if (!options.stream()) {
            return token -> {};
        }

        Integer thinkOpen = tokenizer.getSpecialTokens().get("<think>");
        Integer thinkClose = tokenizer.getSpecialTokens().get("</think>");
        if (thinkOpen == null || thinkClose == null) {
            return token -> { // no think markers in the vocabulary: plain content streaming
                if (!tokenizer.isSpecialToken(token)) {
                    byte[] bytes = tokenizer.decodeTokenBytes(token);
                    System.out.write(bytes, 0, bytes.length);
                }
            };
        }

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
            if (!tokenizer.isSpecialToken(token)) {
                byte[] bytes = tokenizer.decodeTokenBytes(token);
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

    static void runInteractive(Llama model, Sampler sampler, Options options) throws IOException {
        Llama.State state = null;
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> conversationTokens = new ArrayList<>();
        conversationTokens.add(chatFormat.beginOfSentence);
        if (options.systemPrompt() != null) {
            conversationTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
        }
        int startPosition = 0;
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null) break;
                switch (userText) {
                    case "/quit", "/exit" -> { return; }
                    case "/context" -> {
                        System.out.printf("%d out of %d context tokens used (%d tokens remaining)%n",
                                conversationTokens.size(),
                                options.maxTokens(),
                                options.maxTokens() - conversationTokens.size());
                        continue;
                    }
                }
            if (state == null) {
                state = model.createNewState();
            }
            conversationTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.USER, userText)));
            conversationTokens.addAll(chatFormat.encodeGenerationPrompt());
            if (!options.think()) {
                chatFormat.appendThinkSurrogate(conversationTokens);
            }

            List<Integer> promptDelta = conversationTokens.subList(startPosition, conversationTokens.size());
            Engine.GenerationResult result;
            try {
                result = generateCli(model, state, startPosition, promptDelta, chatFormat.getStopTokens(), sampler, options);
            } catch (IllegalArgumentException e) { // the next turn no longer fits the context
                System.err.println("Ran out of context length...");
                break;
            }
            Integer stopToken = result.stopToken() >= 0 ? result.stopToken() : null;
            List<Integer> visibleResponseTokens = Engine.visibleTokens(model.tokenizer(), result.tokens(), options.think());
            conversationTokens.addAll(options.keepPastThinking() ? result.tokens() : visibleResponseTokens);
            if (stopToken != null) {
                conversationTokens.add(stopToken);
                if (stopToken == chatFormat.endOfTurn) {
                    conversationTokens.addAll(model.tokenizer().encode("\n"));
                }
            }
            startPosition = conversationTokens.size();
            if (!options.stream()) {
                System.out.println(result.text());
            }
            if (stopToken == null) {
                System.err.println("Ran out of context length...");
                break;
            }
            }
        }
    }

    static void runInstructOnce(Llama model, Sampler sampler, Options options) {
        Llama.State state = model.createNewState();
        LFMChatFormat chatFormat = new LFMChatFormat(model.tokenizer());
        List<Integer> promptTokens = new ArrayList<>();
        if (options.rawPrompt()) {
            promptTokens.addAll(model.tokenizer().encodeWithSpecialTokens(options.prompt()));
        } else {
            promptTokens.add(chatFormat.beginOfSentence);
            if (options.suffix() != null) {
                promptTokens.addAll(chatFormat.encodeFillInTheMiddle(options.prompt(), options.suffix()));
            } else {
                if (options.systemPrompt() != null) {
                    promptTokens.addAll(chatFormat.encodeSystemThinkingTurn(options.systemPrompt()));
                }
                promptTokens.addAll(chatFormat.encodeMessage(new LFMChatFormat.Message(LFMChatFormat.Role.USER, options.prompt())));
                promptTokens.addAll(chatFormat.encodeGenerationPrompt());
                if (!options.think()) {
                    chatFormat.appendThinkSurrogate(promptTokens);
                }
            }
        }

        Engine.GenerationResult result = generateCli(model, state, 0, promptTokens, chatFormat.getStopTokens(), sampler, options);
        if (!options.stream()) {
            System.out.println(result.text());
        }
    }

    /** One engine pass plus CLI presentation: prompt echo, token streaming through the printer,
     *  and the stderr timing summary line. --max-tokens is a TOTAL context cap in the CLI, so it
     *  is converted to the engine's completion budget here. */
    private static Engine.GenerationResult generateCli(Model model, InferenceState state, int startPosition,
                                                       List<Integer> promptTokens, Set<Integer> stopTokens,
                                                       Sampler sampler, Options options) {
        LFMTokenizer tokenizer = model.tokenizer();
        if (options.echo()) {
            for (int token : promptTokens) {
                System.err.print(replaceControlCharacters(tokenizer.decode(token)));
            }
        }
        IntConsumer printer = streamingPrinter(tokenizer, options);
        IntConsumer onToken = !options.echo() ? printer : token -> {
            System.err.print(replaceControlCharacters(tokenizer.decode(token)));
            printer.accept(token);
        };
        int budget = options.maxTokens() < 0 ? -1
                : options.maxTokens() - Engine.prefillPositions(state, startPosition, promptTokens);
        Engine.Params params = new Engine.Params(sampler, budget, 0, // CLI: no generation deadline
                new Engine.StopSpec(stopTokens, List.of()), options.think());
        Engine.GenerationResult result = Engine.generate(model, state, startPosition, promptTokens, params,
                new Engine.Listener(onToken, null, null, null), GenerationHooks.NONE);
        int generated = result.tokens().size() + (result.stopToken() >= 0 ? 1 : 0);
        String timingPrefix = options.colors() ? ANSI_CYAN : "";
        String timingSuffix = options.colors() ? ANSI_RESET : "";
        System.err.printf("%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + promptTokens.size() + generated, model.contextLength(),
                promptTokens.size() / (result.promptMillis() / 1000.0), promptTokens.size(),
                generated / (result.predictedMillis() / 1000.0), generated,
                timingSuffix);
        return result;
    }

    /** Escape control characters (except newline) so token echo cannot distort the terminal. */
    private static String replaceControlCharacters(String str) {
        StringBuilder chars = new StringBuilder();
        str.codePoints().forEach(cp -> {
            if (Character.getType(cp) == Character.CONTROL && cp != '\n') {
                chars.append("\\u").append(HexFormat.of().toHexDigits(cp, 4));
            } else {
                chars.appendCodePoint(cp);
            }
        });
        return chars.toString();
    }

    static final int DEFAULT_MAX_TOKENS = 1024;

    record Options(Path modelPath, String prompt, String suffix, String systemPrompt, boolean interactive, boolean server, String host, int port,
                    float temperature, float topp, long seed, int maxTokens, boolean stream, boolean echo,
                    boolean think, boolean thinkInline, boolean colors,
                    boolean keepPastThinking, boolean rawPrompt, List<String> warmPrompts,
                    boolean noGrammar) {

        Options {
            require(modelPath != null, "Missing argument: --model <path> is required");
            require(server || interactive || prompt != null, "Missing argument: --prompt is required in --instruct mode e.g. --prompt \"Why is the sky blue?\"");
            require(0 <= temperature, "Invalid argument: --temperature must be non-negative");
            require(0 <= topp && topp <= 1, "Invalid argument: --top-p must be within [0, 1]");
            require(0 <= port && port <= 65535, "Invalid argument: --port must be within [0, 65535]");
        }

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
                    require(false, "Invalid argument for %s: expected true|false|on|off, got %s", optionName, value);
                    yield false;
                }
            };
        }

        static void printUsage(PrintStream out) {
            out.println("Usage:  java -jar lfm25.jar [options]");
            out.println();
            out.println("Options:");
            out.println("  --model, -m <path>            required, path to .gguf file");
            out.println("  --interactive, --chat, -i     run in chat mode");
            out.println("  --instruct                    run in instruct (once) mode, default mode");
            out.println("  --server                      run an OpenAI-compatible HTTP server");
            out.println("  --host <host>                 server bind host, default 127.0.0.1");
            out.println("  --port <int>                  server bind port, default 17325");
            out.println("  --prompt, -p <string>         input prompt");
            out.println("  --suffix <string>             suffix for fill-in-the-middle request");
            out.println("  --system-prompt, -sp <string> system prompt for chat/instruct mode");
            out.println("  --temperature, -temp <float>  temperature in [0,inf], default 1.0");
            out.println("  --top-p <float>               p value in top-p (nucleus) sampling in [0,1] default 0.95");
            out.println("  --seed <long>                 random seed, default System.nanoTime()");
            out.println("  --max-tokens, -n <int>        number of steps to run for < 0 = limited by context length, default " + DEFAULT_MAX_TOKENS);
            out.println("  --stream <boolean>            print tokens during generation; accepts true|false|on|off, default true");
            out.println("  --echo <boolean>              print ALL tokens to stderr; accepts true|false|on|off, default false");
            out.println("  --color <on|off|auto>         colorize thinking output in terminal (default: auto)");
            out.println("  --think <off|on|inline>       on: show thinking (default), off: hide thinking from output (model still generates it), inline: thoughts to stdout");
            out.println("  --keep-past-thinking <bool>   keep prior assistant thinking in history (default false)");
            out.println("  --raw-prompt                  bypass chat template and tokenize --prompt directly");
            out.println("  --warm-prompt <file>          server: pre-ingest the file into the prompt cache (fully");
            out.println("                                dense - requests diverging anywhere inside it resume");
            out.println("                                token-exact); repeatable");
            out.println();
            out.println("Interactive commands:");
            out.println("  /quit, /exit                  exit the chat");
            out.println("  /context                      show context token usage");
            out.println();
            out.println("Examples:");
            out.println("  java -jar lfm25.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat");
            out.println("  java -jar lfm25.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --prompt \"Tell me a joke\"");
            out.println("  java -jar lfm25.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --chat --system-prompt \"You are a helpful assistant\"");
            out.println("  java -jar lfm25.jar --model LFM2.5-1.2B-Instruct-Q8_0.gguf --server --port 17325");
        }

        static Options parseOptions(String[] args) {
            String prompt = null;
            String suffix = null;
            String systemPrompt = null;
            float temperature = 1f;
            float topp = 0.95f;
            Path modelPath = null;
            long seed = System.nanoTime();
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
            boolean keepPastThinking = false;
            boolean rawPrompt = false;
            boolean noGrammar = false;
            List<String> warmPrompts = new ArrayList<>();

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
                            case "--suffix" -> suffix = nextArg;
                            case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                            case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                            case "--top-p" -> topp = Float.parseFloat(nextArg);
                            case "--model", "-m" -> modelPath = Path.of(nextArg);
                            case "--host" -> host = nextArg;
                            case "--port" -> port = Integer.parseInt(nextArg);
                            case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                            case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                            case "--stream" -> stream = parseBooleanOption(optionName, nextArg);
                            case "--echo" -> echo = parseBooleanOption(optionName, nextArg);
                            case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                            case "--keep-past-thinking" -> keepPastThinking = parseBooleanOption(optionName, nextArg);
                            case "--warm-prompt" -> warmPrompts.add(nextArg);
                            case "--think" -> {
                                String thinkMode = nextArg.toLowerCase(Locale.ROOT);
                                thinkInline = List.of("inline", "stdout").contains(thinkMode);
                                switch (thinkMode) {
                                    case "on", "true", "inline", "stdout" -> think = true;
                                    case "off", "false" -> think = false;
                                    default -> require(false, "Invalid argument for %s: expected off|on|inline (or false|true|stdout), got %s", optionName, nextArg);
                                }
                            }
                            default -> require(false, "Unknown option: %s", optionName);
                        }
                    }
                }
            }
            require(List.of("on", "off", "auto").contains(colorMode), "Invalid argument: --color must be one of on|off|auto");
            boolean color = LFM25.supportsAnsiColors(colorMode);
            // server mode: thinking stays on by default (reasoning_content needs it); requests
            // can opt out per-call via chat_template_kwargs.enable_thinking=false
            return new Options(modelPath, prompt, suffix, systemPrompt, interactive, server, host, port, temperature, topp, seed, maxTokens, stream, echo, think, thinkInline, color, keepPastThinking, rawPrompt, List.copyOf(warmPrompts), noGrammar);
        }
    }

    public static void main(String[] args) throws IOException {
        Options options;
        try {
            options = Options.parseOptions(args);
        } catch (IllegalArgumentException e) {
            System.out.println("ERROR " + e.getMessage());
            System.out.println();
            Options.printUsage(System.out);
            System.exit(-1);
            return;
        }
        Model model = AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = ModelLoader.loadModel(options.modelPath(), options.maxTokens());
        }
        if (options.server()) {
            Server.run(model, options);
            return;
        }
        Sampler sampler = Engine.configuredSampler(model, options.think(), options.temperature(), options.topp(), options.seed());
        // LFM2.5 keeps its specialized CLI (FIM, raw-prompt, incremental ChatML, think surrogate);
        // every other architecture runs through the generic chat-format path.
        if (model instanceof Llama llama) {
            if (options.interactive()) {
                runInteractive(llama, sampler, options);
            } else {
                runInstructOnce(llama, sampler, options);
            }
        } else {
            runGeneric(model, sampler, options);
        }
    }

    /** CLI driver for any {@link Model} via its {@link ChatFormat}: a one-shot {@code --prompt} or
     *  an interactive {@code --chat} loop, rebuilding the full prompt each turn (no incremental
     *  resume). Used for non-LFM architectures (e.g. Gemma4). */
    static void runGeneric(Model model, Sampler sampler, Options options) throws IOException {
        ChatFormat chatFormat = model.chatFormat();
        Set<Integer> stops = model.stopTokens();
        if (!options.interactive()) {
            List<Object> messages = new ArrayList<>();
            if (options.systemPrompt() != null) {
                messages.add(Map.of("role", "system", "content", options.systemPrompt()));
            }
            messages.add(Map.of("role", "user", "content", options.prompt()));
            List<Integer> promptTokens = options.rawPrompt()
                    ? new ArrayList<>(model.tokenizer().encodeWithSpecialTokens(options.prompt()))
                    : chatFormat.encode(new ChatContext(messages, null, null, true, options.think(), Map.of()));
            Engine.GenerationResult result = generateCli(model, model.createNewState(), 0, promptTokens, stops, sampler, options);
            if (!options.stream()) {
                System.out.println(result.text());
            }
            return;
        }

        List<Object> history = new ArrayList<>();
        if (options.systemPrompt() != null) {
            history.add(Map.of("role", "system", "content", options.systemPrompt()));
        }
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(System.in))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                history.add(Map.of("role", "user", "content", userText));
                List<Integer> promptTokens = chatFormat.encode(new ChatContext(history, null, null, true, options.think(), Map.of()));
                // fresh state each turn: the generic path re-encodes the whole conversation
                Engine.GenerationResult result = generateCli(model, model.createNewState(), 0, promptTokens, stops, sampler, options);
                if (!options.stream()) {
                    System.out.println(result.text());
                }
                history.add(Map.of("role", "assistant", "content", result.text()));
            }
        }
    }
}

final class AOT {
    // Holds tensor entries + data offset rather than the parsed GGUF: the GGUF object retains the
    // raw vocab/merges metadata strings (~25MiB of image heap) that the tokenizer has already
    // materialized into its own structures.
    record PartialModel(String modelFileName, Llama model, long tensorDataOffset,
                        List<TensorEntry> tensors,
                        Pair<float[], float[]> ropeFreqsSWA, Pair<float[], float[]> ropeFreqsFull) {}

    private static final PartialModel PRELOADED_GGUF = preLoadGGUF(System.getProperty("llama.PreloadGGUF"));

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
                Llama base = ModelLoader.loadModel(null, gguf, LFM25.DEFAULT_MAX_TOKENS, false);
                // Precompute RoPE frequencies at build time (pure Java arrays, survives native-image)
                Llama.Configuration config = base.configuration();
                Pair<float[], float[]> ropeFreqsSWA = RoPE.precomputeFreqsCis(
                        config.contextLength, config.headSizeSWA, config.ropeThetaSWA);
                Pair<float[], float[]> ropeFreqsFull;
                Map<String, GGMLTensorEntry> tmpEntries = ModelLoader.loadTensors(fileChannel, gguf);
                float[] modelRopeFreqs = ModelLoader.ropeFreqFactors(tmpEntries);
                if (modelRopeFreqs != null) {
                    ropeFreqsFull = RoPE.precomputeFreqsCisFromFreqs(
                            config.contextLength, config.headSizeFull, config.ropeTheta, modelRopeFreqs);
                } else {
                    ropeFreqsFull = RoPE.precomputeFreqsCis(
                            config.contextLength, config.headSizeFull, config.ropeTheta);
                }
                return new PartialModel(
                        path.getFileName().toString(), base,
                        gguf.getTensorDataOffset(), List.copyOf(gguf.getTensors()),
                        ropeFreqsSWA, ropeFreqsFull);
            }
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static Llama tryUsePreLoaded(Path modelPath, int contextLength) throws IOException {
        AOT.PartialModel preLoaded = AOT.PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        String optionsModel = modelPath.getFileName().toString();
        String preLoadedModel = preLoaded.modelFileName();
        if (!Objects.equals(optionsModel, preLoadedModel)) {
            return null;
        }
        Llama baseModel = preLoaded.model();
        try (var timer = Timer.log("Load tensors from pre-loaded model");
             var fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            Map<String, GGMLTensorEntry> tensorEntries = ModelLoader.loadTensors(fileChannel, preLoaded.tensorDataOffset(), preLoaded.tensors());
            Llama.Weights weights = ModelLoader.loadWeightsWithRoPE(tensorEntries, baseModel.configuration(),
                    preLoaded.ropeFreqsSWA(), preLoaded.ropeFreqsFull());
            return new Llama(baseModel.configuration().withContextLength(contextLength), baseModel.tokenizer(), weights);
        }
    }
}
