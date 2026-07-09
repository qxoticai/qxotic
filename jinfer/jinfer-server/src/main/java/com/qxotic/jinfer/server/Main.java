// LFM2.5 inference in pure Java
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
import com.qxotic.jinfer.chat.Message;
import com.qxotic.jinfer.chat.TurnTemplate;
import com.qxotic.jinfer.kernels.*;
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

    private static IntConsumer streamingPrinter(GgufTokenizer tokenizer, LLMOptions options) {
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

    /**
     * One generation pass plus CLI presentation: prompt echo, token streaming through the printer,
     * and the stderr timing summary line. --max-tokens is a TOTAL context cap in the CLI, so it is
     * converted to the generator's completion budget here.
     */
    private static <S extends RuntimeState> Generator.GenerationResult generateCli(
            LanguageModel<?, ?, S> model,
            S state,
            List<Integer> promptTokens,
            Set<Integer> stopTokens,
            Sampler sampler,
            LLMOptions options) {
        GgufTokenizer tokenizer = model.tokenizer();
        if (options.echo()) {
            for (int token : promptTokens) {
                System.err.print(replaceControlCharacters(tokenizer.decode(token)));
            }
        }
        IntConsumer printer = streamingPrinter(tokenizer, options);
        IntConsumer onToken =
                !options.echo()
                        ? printer
                        : token -> {
                            System.err.print(replaceControlCharacters(tokenizer.decode(token)));
                            printer.accept(token);
                        };
        int startPosition = state.position();
        int budget =
                options.maxTokens() < 0
                        ? -1
                        : options.maxTokens() - (startPosition + promptTokens.size());
        // Chunk long prompts to the state's batch capacity; the final chunk rides through the
        // Generator (which needs >=1 prompt token for fresh logits).
        int totalPrompt = promptTokens.size();
        long chunkNanos = 0;
        int cap = state.batchCapacity();
        while (promptTokens.size() > cap) {
            int[] ids = new int[cap];
            for (int i = 0; i < cap; i++) ids[i] = promptTokens.get(i);
            if (options.echo()) {
                for (int id : ids) System.err.print(replaceControlCharacters(tokenizer.decode(id)));
            }
            long t0 = System.nanoTime();
            model.ingest(state, Batch.prefill(ids));
            chunkNanos += System.nanoTime() - t0;
            promptTokens = promptTokens.subList(cap, promptTokens.size());
        }
        Generator.Params params =
                new Generator.Params(
                        sampler,
                        budget,
                        0, // CLI: no generation deadline
                        new Generator.StopSpec(stopTokens, List.of()),
                        options.think());
        Generator.GenerationResult result =
                Generator.generate(
                        model,
                        state,
                        promptTokens,
                        params,
                        new Generator.Listener(onToken, null, null, null));
        int generated = result.tokens().size() + (result.stopToken() >= 0 ? 1 : 0);
        String timingPrefix = options.colors() ? ANSI_CYAN : "";
        String timingSuffix = options.colors() ? ANSI_RESET : "";
        System.err.printf(
                "%n%scontext: %d/%d prompt: %.2f tokens/s (%d) generation: %.2f tokens/s (%d)%s%n",
                timingPrefix,
                startPosition + totalPrompt + generated,
                model.config().contextLength(),
                totalPrompt / (chunkNanos / 1e6 / 1000.0 + result.promptMillis() / 1000.0),
                totalPrompt,
                generated / (result.predictedMillis() / 1000.0),
                generated,
                timingSuffix);
        return result;
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
        out.println(
                "  --top-p <float>               p value in top-p (nucleus) sampling in [0,1]"
                        + " default 0.95");
        out.println("  --seed <long>                 random seed, default System.nanoTime()");
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
                "  --keep-past-thinking <bool>   keep prior assistant thinking in history (default"
                        + " false)");
        out.println(
                "  --raw-prompt                  bypass chat template and tokenize --prompt"
                        + " directly");
        out.println(
                "  --sealed <file>               instruct: sealed-prompt cache for the"
                        + " (system+prompt) prefix;");
        out.println(
                "                                created on first run, restored (instant TTFT)"
                        + " after");
        out.println(
                "  --warm-prompt <file>          server: pre-ingest the file into the prompt cache"
                        + " (fully");
        out.println(
                "                                dense - requests diverging anywhere inside it"
                        + " resume");
        out.println("                                token-exact); repeatable");
        out.println();
        out.println("Interactive commands:");
        out.println("  /quit, /exit                  exit the chat");
        out.println("  /context                      show context token usage");
        out.println();
        out.println("Examples:");
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
        Path sealedPrompt = null;
        List<String> warmPrompts = new ArrayList<>();

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
                        case "--suffix" -> suffix = nextArg;
                        case "--system-prompt", "-sp" -> systemPrompt = nextArg;
                        case "--temperature", "--temp" -> temperature = Float.parseFloat(nextArg);
                        case "--top-p" -> topp = Float.parseFloat(nextArg);
                        case "--model", "-m" -> modelPath = Path.of(nextArg);
                        case "--host" -> host = nextArg;
                        case "--port" -> port = Integer.parseInt(nextArg);
                        case "--seed", "-s" -> seed = Long.parseLong(nextArg);
                        case "--max-tokens", "-n" -> maxTokens = Integer.parseInt(nextArg);
                        case "--stream" ->
                                stream = LLMOptions.parseBooleanOption(optionName, nextArg);
                        case "--echo" -> echo = LLMOptions.parseBooleanOption(optionName, nextArg);
                        case "--color" -> colorMode = nextArg.toLowerCase(Locale.ROOT);
                        case "--keep-past-thinking" ->
                                keepPastThinking =
                                        LLMOptions.parseBooleanOption(optionName, nextArg);
                        case "--warm-prompt" -> warmPrompts.add(nextArg);
                        case "--sealed" -> sealedPrompt = Path.of(nextArg);
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
        return new LLMOptions(
                modelPath,
                prompt,
                suffix,
                systemPrompt,
                interactive,
                server,
                host,
                port,
                temperature,
                topp,
                seed,
                maxTokens,
                stream,
                echo,
                think,
                thinkInline,
                color,
                keepPastThinking,
                rawPrompt,
                List.copyOf(warmPrompts),
                noGrammar,
                sealedPrompt);
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

    private static PrintStream utf8Stream(FileDescriptor fd) {
        return new PrintStream(
                new BufferedOutputStream(new FileOutputStream(fd), 8192),
                true,
                StandardCharsets.UTF_8);
    }

    public static void main(String[] args) throws IOException {
        forceUtf8Console();
        LLMOptions options;
        try {
            options = parseOptions(args);
        } catch (IllegalArgumentException e) {
            System.out.println("ERROR " + e.getMessage());
            System.out.println();
            printUsage(System.out);
            System.exit(-1);
            return;
        }
        LanguageModel<?, ?, ?> model =
                AOT.tryUsePreLoaded(options.modelPath(), options.maxTokens());
        if (model == null) {
            model = Models.load(options.modelPath(), options.maxTokens());
        }
        if (options.server()) {
            Server.start(model, options);
            return;
        }
        Sampler sampler =
                Generator.configuredSampler(
                        model,
                        options.think(),
                        options.temperature(),
                        options.topp(),
                        options.seed());
        runGeneric(model, sampler, options);
    }

    /**
     * CLI driver: a one-shot {@code --prompt} or an interactive {@code --chat} loop. Models
     * exposing a {@link TurnTemplate} run INCREMENTALLY - one persistent state, only each new turn
     * ingested (the validated turn-by-turn framing; replies stay in the KV verbatim); everything
     * else falls back to the whole-render Jinja path with a fresh state per turn.
     */
    static <S extends RuntimeState> void runGeneric(
            LanguageModel<?, ?, S> model, Sampler sampler, LLMOptions options) throws IOException {
        TurnTemplate template = options.rawPrompt() ? null : model.turnTemplate().orElse(null);
        if (!options.interactive()) {
            runInstruct(model, template, sampler, options);
        } else if (template != null) {
            runChatIncremental(model, template, sampler, options);
        } else {
            runChatWholeRender(model, sampler, options);
        }
    }

    private static <S extends RuntimeState> void runInstruct(
            LanguageModel<?, ?, S> model,
            TurnTemplate template,
            Sampler sampler,
            LLMOptions options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        List<Integer> promptTokens;
        if (options.rawPrompt()) {
            promptTokens =
                    new ArrayList<>(model.tokenizer().encodeWithSpecialTokens(options.prompt()));
        } else if (template != null) {
            List<Message> turns = new ArrayList<>();
            if (options.systemPrompt() != null) {
                turns.add(Message.system(options.systemPrompt()));
            }
            turns.add(Message.user(options.prompt()));
            List<Batch> batches = new ArrayList<>(template.encode(template.normalize(turns)));
            batches.addAll(template.generationPrompt(options.think()));
            promptTokens = new ArrayList<>();
            for (int id : Batch.tokenIds(batches)) promptTokens.add(id);
        } else {
            List<Object> messages = new ArrayList<>();
            if (options.systemPrompt() != null) {
                messages.add(Map.of("role", "system", "content", options.systemPrompt()));
            }
            messages.add(Map.of("role", "user", "content", options.prompt()));
            promptTokens =
                    ChatFormat.encode(
                            model.tokenizer(),
                            new ChatContext(messages, null, true, options.think(), Map.of()));
        }
        S state =
                model.newState(
                        model.config().contextLength(),
                        Math.min(
                                Math.max(promptTokens.size(), 16),
                                RuntimeFlags.BATCH_CAPACITY)); // ports chunk long prompts

        // --sealed: restore the (system+prompt) prefix from the sealed file (instant TTFT), or
        // create it
        // on the first run. The seal covers every position BUT THE LAST: it captures the
        // KV/recurrent
        // state, not the transient last-position activations (s.residual / lastChunkLen) that
        // logits()
        // reads. So the final token is always ingested fresh, and its forward pass reproduces those
        // — for
        // attention and recurrent layers alike (the last token also advances the conv state from
        // as-of-P-1
        // to as-of-P). Any mismatch (different prompt/model) falls back to a plain prefill.
        if (options.sealedPrompt() != null
                && model.stateCodec().isPresent()
                && promptTokens.size() >= 2) {
            var codec = model.stateCodec().get();
            int last = promptTokens.get(promptTokens.size() - 1);
            long[] fp = new long[promptTokens.size() - 1];
            for (int i = 0; i < fp.length; i++) fp[i] = promptTokens.get(i);
            byte[] seed = com.qxotic.jinfer.cache.PromptCache.modelSeed(options.modelPath());
            if (Files.exists(options.sealedPrompt())) {
                long t0 = System.nanoTime();
                int restored =
                        com.qxotic.jinfer.cache.SealedPrompt.open(options.sealedPrompt(), seed)
                                .tryRestore(state, codec, fp);
                if (restored == fp.length) {
                    System.err.printf(
                            "sealed prompt restored: %d positions in %.1f ms%n",
                            restored, (System.nanoTime() - t0) / 1e6);
                    promptTokens =
                            List.of(last); // re-ingest the last token to rebuild the hidden state
                } else {
                    System.err.println("sealed prompt mismatch: prefilling normally");
                }
            } else {
                model.ingest(state, Batch.prefill(fpToIds(fp)));
                com.qxotic.jinfer.cache.SealedPrompt.compile(
                        options.sealedPrompt(), "cli", codec, state, fp, seed);
                System.err.println(
                        "sealed prompt compiled: "
                                + options.sealedPrompt()
                                + " ("
                                + fp.length
                                + " positions)");
                promptTokens = List.of(last);
            }
        }

        Generator.GenerationResult result =
                generateCli(model, state, promptTokens, stops, sampler, options);
        if (!options.stream()) {
            System.out.println(result.text());
        }
    }

    private static int[] fpToIds(long[] fp) {
        int[] ids = new int[fp.length];
        for (int i = 0; i < fp.length; i++) ids[i] = (int) fp[i];
        return ids;
    }

    /**
     * Interactive chat on the model's TurnTemplate: ONE persistent state; per turn only the delta
     * is ingested - closeTurn (re-framing the un-ingested stop token) + the user turn + the
     * generation prompt. The reply tokens are already in the KV from decoding.
     */
    private static <S extends RuntimeState> void runChatIncremental(
            LanguageModel<?, ?, S> model,
            TurnTemplate template,
            Sampler sampler,
            LLMOptions options)
            throws IOException {
        Set<Integer> stops = model.stopTokens();
        S state = model.newState(model.config().contextLength(), RuntimeFlags.BATCH_CAPACITY);
        boolean first = true;
        try (BufferedReader reader =
                new BufferedReader(new InputStreamReader(System.in, StandardCharsets.UTF_8))) {
            while (true) {
                System.out.print("> ");
                System.out.flush();
                String userText = reader.readLine();
                if (userText == null || "/quit".equals(userText) || "/exit".equals(userText)) break;
                if ("/context".equals(userText)) {
                    System.out.printf(
                            "context: %d/%d tokens%n",
                            state.position(), model.config().contextLength());
                    continue;
                }
                List<Batch> batches = new ArrayList<>();
                List<Message> turns = new ArrayList<>();
                if (first) {
                    batches.addAll(template.conversationStart());
                    if (options.systemPrompt() != null) {
                        turns.add(Message.system(options.systemPrompt()));
                    }
                } else {
                    batches.addAll(template.closeTurn()); // close the previous assistant turn
                }
                turns.add(Message.user(userText));
                for (Message m : first ? template.normalize(turns) : turns) {
                    batches.addAll(template.encodeTurn(m));
                }
                batches.addAll(template.generationPrompt(options.think()));
                List<Integer> delta = new ArrayList<>();
                for (int id : Batch.tokenIds(batches)) delta.add(id);
                Generator.GenerationResult result =
                        generateCli(model, state, delta, stops, sampler, options);
                if (!options.stream()) {
                    System.out.println(result.text());
                }
                first = false;
            }
        }
    }

    /**
     * Whole-render fallback for models without a TurnTemplate: re-encode the full conversation
     * through the Jinja template each turn, fresh state.
     */
    private static <S extends RuntimeState> void runChatWholeRender(
            LanguageModel<?, ?, S> model, Sampler sampler, LLMOptions options) throws IOException {
        Set<Integer> stops = model.stopTokens();
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
                List<Integer> promptTokens =
                        ChatFormat.encode(
                                model.tokenizer(),
                                new ChatContext(history, null, true, options.think(), Map.of()));
                Generator.GenerationResult result =
                        generateCli(
                                model,
                                model.newState(
                                        model.config().contextLength(),
                                        Math.min(
                                                Math.max(promptTokens.size(), 16),
                                                RuntimeFlags.BATCH_CAPACITY)),
                                promptTokens,
                                stops,
                                sampler,
                                options);
                if (!options.stream()) {
                    System.out.println(result.text());
                }
                history.add(Map.of("role", "assistant", "content", result.text()));
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
    static LanguageModel<?, ?, ?> tryUsePreLoaded(Path modelPath, int contextLength)
            throws IOException {
        PartialModel preLoaded = PRELOADED_GGUF;
        if (preLoaded == null) {
            return null;
        }
        if (!Objects.equals(modelPath.getFileName().toString(), preLoaded.modelFileName())) {
            return null;
        }
        try (var timer = Timer.log("Load tensors from pre-loaded model");
                FileChannel fileChannel = FileChannel.open(modelPath, StandardOpenOption.READ)) {
            return Models.load(fileChannel, preLoaded.gguf(), contextLength);
        }
    }
}
