package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampling;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AccessDeniedException;
import java.nio.file.FileSystemException;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.function.Supplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * One parsing pass, followed by validation. Resolution and runtime setup are explicit operations.
 */
final class Options {
    static final Set<String> MODEL_COMMANDS =
            Set.of("chat", "instruct", "server", "speak", "transcribe");
    static final Set<String> TEXT_COMMANDS = Set.of("chat", "instruct", "server");
    static final Set<String> CONVERSATION_COMMANDS = Set.of("chat", "instruct");

    String command;
    boolean help;
    boolean version;
    String input;
    final List<String> operands = new ArrayList<>();

    String modelRef;
    String tokenizerRef;
    final Map<String, String> companionRefs = new LinkedHashMap<>();
    Integer threads;
    Integer batchCapacity;
    Integer contextCapacity;
    String systemPrompt;
    Float temperature;
    Float topP;
    Integer topK;
    Float minP;
    Long seed;
    int maxOutputTokens = -1;
    boolean think = true;
    boolean thinkInline;
    Integer maxReasoningTokens;
    String reasoningCutoffMessage;
    int speculationDepth = 4;
    Path promptCache;
    boolean promptCacheReadOnly;
    Boolean stream;
    boolean echo;
    boolean rawPrompt;
    String color = "auto";

    final Server.Settings server = new Server.Settings();
    final Speak.Settings speech = new Speak.Settings();
    final Transcribe.Settings transcription = new Transcribe.Settings();
    boolean force;

    // Presence matters: an explicitly supplied, inapplicable default is still a mistake.
    private final Map<String, Set<String>> used = new LinkedHashMap<>();

    private Options() {}

    static Options parse(String... argv) {
        Options o = new Options();
        Args args = new Args(argv);
        try {
            IllegalArgumentException firstError = null;
            while (args.next()) {
                try {
                    o.read(args);
                } catch (IllegalArgumentException failure) {
                    // Finish consuming tokens so a trailing --help can explain a bad invocation.
                    // Option values and everything after -- remain literal, including "--help".
                    if (firstError == null) firstError = failure;
                }
            }
            if (argv.length == 0) o.help = true;
            if (o.help || o.version) return o;
            if (firstError != null) throw firstError;
            if (o.command == null) throw o.usageHelp("missing command");
            o.validate();
            return o;
        } catch (IllegalArgumentException failure) {
            UsageException usage =
                    failure instanceof UsageException u
                            ? u
                            : new UsageException(failure.getMessage());
            usage.command = o.command;
            throw usage;
        }
    }

    private void read(Args args) {
        if (!args.isOption()) {
            if (command != null) operands.add(args.token);
            else select(args.token);
            return;
        }
        switch (args.name) {
            case "--help", "-h" -> help = args.flag();
            case "--version" -> version = args.flag();
            default -> {
                if (readShared(args)) return;
                if (!(Instruct.read(this, args)
                        || Server.read(this, args)
                        || Speak.read(this, args)
                        || Transcribe.read(this, args)
                        || Hub.read(this, args))) throw usageHelp("unknown option: " + args.name);
                require(command != null, "%s must follow the command", args.name);
            }
        }
    }

    private void select(String name) {
        String selected =
                switch (name) {
                    case "serve" -> "server";
                    case "prompt" -> "instruct";
                    default -> name;
                };
        if (!MODEL_COMMANDS.contains(selected)
                && !Set.of("pull", "list", "cache-info").contains(selected))
            throw usageHelp("unknown command: " + name);
        command = selected;
    }

    private UsageException usageHelp(String message) {
        UsageException error = new UsageException(message);
        error.showHelp = true;
        return error;
    }

    private boolean readModel(Args a) {
        switch (a.name) {
            case "--model", "-m" -> modelRef = a.value();
            case "--with" -> {
                String attached = a.value();
                int eq = attached.indexOf('=');
                require(
                        eq > 0 && eq < attached.length() - 1,
                        "--with takes <role>=<path|ref>, got %s",
                        attached);
                attach(attached.substring(0, eq), attached.substring(eq + 1));
            }
            case "--threads", "-t" -> threads = a.integer();
            case "--color" -> {
                color = a.value().toLowerCase(Locale.ROOT);
                use(a.name, Set.of("chat", "instruct", "transcribe")); // read by Turn and the HUD
                return true;
            }
            default -> {
                return false;
            }
        }
        use(a.name, MODEL_COMMANDS);
        return true;
    }

    private void attach(String role, String value) {
        require(
                !role.isBlank() && !value.isBlank(),
                "companions require a non-blank role and reference");
        require(!value.equals("auto"), "name the '%s' file explicitly instead of 'auto'", role);
        require(!role.equals("model"), "select the model with --model <path|ref>, not --with");
        switch (role) {
            case "tokenizer" -> {
                tokenizerRef = value;
                use("--with tokenizer", TEXT_COMMANDS);
            }
            default -> {
                require(!companionRefs.containsKey(role), "companion given twice: %s", role);
                companionRefs.put(role, value);
            }
        }
    }

    private boolean readGeneration(Args a) {
        switch (a.name) {
            case "--batch-capacity" -> batchCapacity = a.integer();
            case "--context-capacity", "-c" -> contextCapacity = a.integer();
            case "--temp" -> temperature = a.decimal();
            case "--top-p" -> topP = a.decimal();
            case "--top-k" -> topK = a.integer();
            case "--min-p" -> minP = a.decimal();
            case "--seed", "-s" -> seed = a.longValue();
            case "--max-output-tokens", "-n" -> maxOutputTokens = a.integer();
            case "--speculation-depth" -> speculationDepth = a.integer();
            case "--max-reasoning-tokens" -> maxReasoningTokens = a.integer();
            case "--reasoning-cutoff-message" -> reasoningCutoffMessage = a.value();
            case "--think" -> {
                String mode = a.value().toLowerCase(Locale.ROOT);
                thinkInline = mode.equals("inline");
                think =
                        switch (mode) {
                            case "on", "inline" -> true;
                            case "off" -> false;
                            default ->
                                    throw new UsageException(
                                            "--think expects off|on|inline, got " + mode);
                        };
            }
            default -> {
                return false;
            }
        }
        use(a.name, TEXT_COMMANDS);
        return true;
    }

    private boolean readShared(Args a) {
        if (readModel(a) || readGeneration(a)) return true;
        Set<String> commands =
                switch (a.name) {
                    case "--system-prompt" -> {
                        systemPrompt = a.value();
                        yield CONVERSATION_COMMANDS;
                    }
                    case "--stream" -> {
                        stream = a.flag();
                        yield Set.of("chat", "instruct", "speak");
                    }
                    case "--no-stream" -> {
                        a.flag();
                        stream = false;
                        yield Set.of("chat", "instruct", "speak");
                    }
                    case "--echo" -> {
                        echo = a.flag();
                        yield CONVERSATION_COMMANDS;
                    }
                    case "--no-echo" -> {
                        a.flag();
                        echo = false;
                        yield CONVERSATION_COMMANDS;
                    }
                    case "--cache", "--cache-ro" -> {
                        promptCache = Path.of(a.value());
                        promptCacheReadOnly = a.name.equals("--cache-ro");
                        yield Set.of("instruct", "server");
                    }
                    default -> null;
                };
        if (commands == null) return false;
        use(a.name, commands);
        return true;
    }

    void use(String flag, Set<String> commands) {
        used.put(flag, commands);
    }

    boolean supplied(String flag) {
        return used.containsKey(flag);
    }

    /**
     * chat and instruct send text: a media projector would load (gigabytes, for Gemma's) and never
     * see an image, and the model would say it cannot see the file it was "given".
     */
    void rejectMedia() {
        require(
                !companionRefs.containsKey("media"),
                "--with media would load a projector %s never uses, as it sends text only; serve"
                        + " the model with 'jinfer server' and send images or audio in"
                        + " /v1/chat/completions",
                command);
    }

    /** Refuses every language option given, except the {@code kept} ones the application uses. */
    void rejectLanguageOptions(String application, String... kept) {
        for (var option : used.entrySet()) {
            require(
                    !option.getValue().equals(TEXT_COMMANDS)
                            || List.of(kept).contains(option.getKey()),
                    "%s does not apply to %s",
                    option.getKey(),
                    application);
        }
    }

    private void validate() {
        for (var option : used.entrySet())
            require(
                    option.getValue().contains(command),
                    "%s does not apply to %s",
                    option.getKey(),
                    command);
        if (MODEL_COMMANDS.contains(command)) {
            require(
                    modelRef != null && !modelRef.isBlank(),
                    "missing model; specify --model <path|ref>");
            require(operands.size() <= 1, "too many inputs; quote text containing spaces");
            if (!operands.isEmpty()) input = operands.getFirst();
        }
        require(threads == null || threads >= 1, "--threads must be at least 1; got %s", threads);
        require(
                batchCapacity == null || batchCapacity >= 1,
                "--batch-capacity must be at least 1; got %s",
                batchCapacity);
        require(
                contextCapacity == null || contextCapacity >= 0,
                "--context-capacity must be non-negative (0 uses the model maximum); got %s",
                contextCapacity);
        require(
                temperature == null || temperature >= 0,
                "--temp must be non-negative; got %s",
                temperature);
        require(
                topP == null || (topP > 0 && topP <= 1),
                "--top-p must be greater than 0 and at most 1; got %s",
                topP);
        require(topK == null || topK >= 0, "--top-k must be non-negative; got %s", topK);
        require(
                minP == null || (minP >= 0 && minP <= 1),
                "--min-p must be within [0, 1]; got %s",
                minP);
        require(
                maxOutputTokens >= -1,
                "--max-output-tokens must be -1 or non-negative; got %s",
                maxOutputTokens);
        require(
                maxReasoningTokens == null || maxReasoningTokens >= -1,
                "--max-reasoning-tokens must be -1 or non-negative; got %s",
                maxReasoningTokens);
        require(
                speculationDepth >= 0 && speculationDepth <= 8,
                "--speculation-depth must be within [0, 8]; got %s",
                speculationDepth);
        require(
                Set.of("auto", "on", "off").contains(color),
                "--color must be one of on|off|auto; got '%s'",
                color);
        if (rawPrompt) {
            require(
                    systemPrompt == null
                            && maxReasoningTokens == null
                            && reasoningCutoffMessage == null
                            && !supplied("--think"),
                    "--raw-prompt bypasses the template; --system-prompt, --think and reasoning"
                            + " budgets cannot apply");
            require(
                    promptCache == null || promptCacheReadOnly,
                    "--cache with --raw-prompt cannot append; use --cache-ro");
        }
        if (stream == null) stream = command.equals("chat") || command.equals("instruct");
        switch (command) {
            case "chat" -> Chat.validate(this);
            case "instruct" -> Instruct.validate(this);
            case "server" -> Server.validate(this);
            case "speak" -> Speak.validate(this);
            case "transcribe" -> Transcribe.validate(this);
            default -> Hub.validate(this);
        }
    }

    /** Runs before anything touches RuntimeFlags: the properties must land first. */
    void configureRuntime() throws IOException {
        if (System.getProperty("org.graalvm.nativeimage.imagecode") == null
                && ModuleLayer.boot().findModule("jdk.incubator.vector").isEmpty())
            throw new IOException(
                    "Run java --add-modules jdk.incubator.vector"
                            + " --enable-native-access=ALL-UNNAMED -jar jinfer.jar ...");
        if (threads != null) System.setProperty("jinfer.threads", threads.toString());
        if (batchCapacity != null)
            System.setProperty("jinfer.batchCapacity", batchCapacity.toString());
    }

    record Files(Path model, Map<String, Path> companions, Path tokenizer) {}

    Files resolve(ModelStore store) throws IOException {
        List<String> refs = new ArrayList<>();
        refs.add(modelRef);
        if (tokenizerRef != null) refs.add(tokenizerRef);
        refs.addAll(companionRefs.values());
        List<Path> paths = resolveFiles(store, refs);
        int at = 0;
        Path model = paths.get(at++);
        Path tokenizer = tokenizerRef == null ? null : paths.get(at++);
        Map<String, Path> companions = new LinkedHashMap<>();
        if (!companionRefs.isEmpty()) {
            Map<String, String> offered = AOT.companionFiles(model);
            for (String role : companionRefs.keySet()) {
                if (!offered.containsKey(role))
                    throw new IOException(
                            "This model has no '"
                                    + role
                                    + "' capability; it offers "
                                    + offered.keySet());
                companions.put(role, paths.get(at++));
            }
        }
        return new Files(model, Map.copyOf(companions), tokenizer);
    }

    static List<Path> resolveFiles(ModelStore store, List<String> refs) throws IOException {
        return resolveFiles(refs, () -> store.resolveAll(refs));
    }

    static List<Path> pullFiles(ModelStore store, List<String> refs, boolean force)
            throws IOException {
        return resolveFiles(refs, () -> store.pullAll(refs, force));
    }

    private static List<Path> resolveFiles(List<String> refs, Supplier<List<Path>> resolve)
            throws IOException {
        for (String ref : refs) {
            String lower = ref.toLowerCase(Locale.ROOT);
            if (ModelStore.isRef(ref)
                    || lower.startsWith("http://")
                    || lower.startsWith("https://")) continue;
            requireFile(Path.of(ref));
        }
        return resolve.get();
    }

    /** Every local input the CLI reads is checked here first, so a missing one reads the same. */
    static void requireFile(Path path) throws IOException {
        if (java.nio.file.Files.isDirectory(path))
            throw new IOException("expected a file, got a directory: '" + path + "'");
        if (!java.nio.file.Files.isRegularFile(path)) throw new IOException(noSuchFile(path));
    }

    private static String noSuchFile(Object path) {
        return "no such file: '" + path + "'";
    }

    Sampling sampling(LoadedModel.SamplingDefaults defaults) {
        return defaults.resolve(temperature, topP, topK, minP, seed);
    }

    PromptCache.Options cacheOptions() {
        var cache = PromptCache.Options.DEFAULTS.withCatalog(promptCache, promptCacheReadOnly);
        return contextCapacity == null ? cache : cache.withContextCapacity(contextCapacity);
    }

    boolean colors(Main.IO io, int fd) {
        if (color.equals("on")) return true;
        return !color.equals("off")
                && io.isTerminal(fd)
                && System.getenv().getOrDefault("NO_COLOR", "").isEmpty()
                && !"dumb".equals(System.getenv("TERM"));
    }

    static final class UsageException extends IllegalArgumentException {
        String command;
        boolean showHelp;

        UsageException(String message) {
            super(message);
        }
    }

    static void require(boolean condition, String message, Object... args) {
        if (!condition) throw new UsageException(message.formatted(args));
    }

    static String rootMessage(Throwable failure) {
        while (failure.getCause() != null
                && (failure.getMessage() == null
                        || failure.getMessage().equals(failure.getCause().toString())))
            failure = failure.getCause();
        if (failure instanceof UncheckedIOException e)
            return e.getMessage() + ": " + rootMessage(e.getCause());
        // NIO's message is the bare path when the OS gave no reason; say what went wrong with it
        if (failure instanceof FileSystemException e && e.getReason() == null)
            return switch (e) {
                case NoSuchFileException x -> noSuchFile(e.getFile());
                case AccessDeniedException x -> "permission denied: '" + e.getFile() + "'";
                default -> e.getFile() + ": " + e.getClass().getSimpleName();
            };
        return failure.getMessage() == null
                ? failure.getClass().getSimpleName()
                : failure.getMessage();
    }

    /** Shared token cursor. Only the reader of a recognized option may consume its value. */
    static final class Args {
        private final String[] argv;
        private int index;
        private boolean literal;
        String token;
        String name;
        String inline;

        Args(String[] argv) {
            this.argv = argv;
        }

        boolean next() {
            if (index == argv.length) return false;
            token = argv[index++];
            if (!literal && token.equals("--")) {
                literal = true;
                return next();
            }
            name = null;
            inline = null;
            if (!literal && token.startsWith("-") && !token.equals("-")) {
                int eq = token.indexOf('=');
                name = eq < 0 ? token : token.substring(0, eq);
                if (eq >= 0) inline = token.substring(eq + 1);
            }
            return true;
        }

        boolean isOption() {
            return name != null;
        }

        String value() {
            if (inline != null) return inline;
            require(index < argv.length, "missing argument for option %s", name);
            return argv[index++];
        }

        boolean flag() {
            require(inline == null, "%s takes no value; got '%s'", name, inline);
            return true;
        }

        int integer() {
            String value = value();
            try {
                return Integer.parseInt(value);
            } catch (NumberFormatException e) {
                throw new UsageException(name + " expects an integer, got " + value);
            }
        }

        long longValue() {
            String value = value();
            try {
                return Long.parseLong(value);
            } catch (NumberFormatException e) {
                throw new UsageException(name + " expects an integer, got " + value);
            }
        }

        float decimal() {
            String value = value();
            try {
                float number = Float.parseFloat(value);
                require(Float.isFinite(number), "%s must be finite; got '%s'", name, value);
                return number;
            } catch (NumberFormatException e) {
                throw new UsageException(name + " expects a number, got " + value);
            }
        }
    }

    static void printUsage(PrintStream out) {
        out.println(
                """
                jinfer - local model inference

                Usage: jinfer [model options] <command> [options] [input]

                  chat          have a conversation
                  instruct      generate one response (alias: prompt)
                  server        serve an HTTP API (alias: serve)
                  speak         turn text into speech
                  transcribe    turn audio into text
                  pull          download model files
                  list          show cached models
                  cache-info    inspect a prompt cache

                Examples:
                  jinfer chat -m model.gguf
                  jinfer -m model.gguf --temp 0.3 instruct "Hello."
                  jinfer speak -m inflect.gguf "Hello world."

                Run 'jinfer <command> --help' for details. --version prints the version.
                JVM: java --add-modules jdk.incubator.vector --enable-native-access=ALL-UNNAMED \\
                       -jar jinfer.jar ...
                """);
    }

    void printHelp(PrintStream out) {
        if (command == null) {
            printUsage(out);
            return;
        }
        var text = new ByteArrayOutputStream();
        var help = new PrintStream(text, true, StandardCharsets.UTF_8);
        switch (command) {
            case "chat" -> Chat.printHelp(help);
            case "instruct" -> Instruct.printHelp(help);
            case "server" -> Server.printHelp(help);
            case "speak" -> Speak.printHelp(help);
            case "transcribe" -> Transcribe.printHelp(help);
            default -> Hub.printHelp(command, help);
        }
        out.print(layout(text.toString(StandardCharsets.UTF_8)));
    }

    /** Where every option's description starts, on every help screen. */
    static final int HELP_COLUMN = 33;

    static final int HELP_WIDTH = 100;

    // "  -x, --option <arg>  description": two spaces or more end the option's spelling
    private static final Pattern OPTION_ROW = Pattern.compile("  (-\\S.*?)(?: {2,}(\\S.*))?");

    /**
     * One description column for a whole help screen written by several classes: each option's
     * description moves to {@link #HELP_COLUMN}, below the option when it is too long, and wraps at
     * {@link #HELP_WIDTH}.
     */
    static String layout(String help) {
        StringBuilder out = new StringBuilder();
        String pad = " ".repeat(HELP_COLUMN);
        boolean inRow = false;
        for (String line : help.lines().toList()) {
            Matcher row = OPTION_ROW.matcher(line);
            if (row.matches()) {
                inRow = true;
                String option = "  " + row.group(1);
                out.append(option);
                if (row.group(2) != null)
                    out.append(
                                    option.length() + 2 > HELP_COLUMN
                                            ? "\n" + pad
                                            : " ".repeat(HELP_COLUMN - option.length()))
                            .append(wrap(row.group(2)));
            } else if (inRow && line.startsWith("      ") && !line.isBlank()) {
                out.append(pad).append(wrap(line.strip()));
            } else {
                inRow = false;
                out.append(line);
            }
            out.append('\n');
        }
        return out.toString();
    }

    /** {@code text} from {@link #HELP_COLUMN}, broken between words at {@link #HELP_WIDTH}. */
    private static String wrap(String text) {
        StringBuilder out = new StringBuilder();
        int column = HELP_COLUMN;
        for (String word : text.split(" ")) {
            if (column > HELP_COLUMN) {
                boolean full = column + 1 + word.length() > HELP_WIDTH;
                out.append(full ? "\n" + " ".repeat(HELP_COLUMN) : " ");
                column = full ? HELP_COLUMN : column + 1;
            }
            out.append(word);
            column += word.length();
        }
        return out.toString();
    }

    static void modelHelp(PrintStream out) {
        out.println(
                """
                Model options (before or after the command):
                  -m, --model <path|ref>       model file or hub reference; required
                  --with <role>=<path|ref>     attach a companion; repeatable for different roles
                  -t, --threads <int>          compute workers (default: physical/fast cores)
                                               overrides -Djinfer.threads when both are supplied

                References: [host/]owner/repo[@revision][/file][:quant]. Default host: hf.co.
                Existing local files win. Remote files are downloaded once and cached.
                Cache: JINFER_MODELS or the platform cache.
                Offline: JINFER_OFFLINE=1|true|on|yes prevents fetching (case-insensitive).
                0|false|off|no disables it; -Djinfer.offline takes the same values and wins.
                """);
    }

    static void generationHelp(PrintStream out) {
        out.println(
                """
                Language-model options (before or after the command):
                  --temp <number>             temperature; 0 selects greedy generation
                  --top-p <number>            nucleus mass in (0, 1]
                  --top-k <int>               candidate limit; 0 disables
                  --min-p <number>            relative probability floor in [0, 1]
                  -s, --seed <long>           sampling seed
                  -c, --context-capacity <int>  state capacity; default min(4096, model); 0: model maximum
                  --batch-capacity <int>      default prefill/scratch width (runtime default: 512)
                  -n, --max-output-tokens <int>  generated-token budget; -1: remaining context
                  --think <off|on>            off: do not reason; on: allow model reasoning
                  --max-reasoning-tokens <int>  reasoning budget; -1: uncapped
                  --reasoning-cutoff-message <text>  forced text when the reasoning budget runs out
                  --speculation-depth <int>   draft depth in [0, 8]; default 4
                  --with tokenizer=<path|ref>  use another GGUF's tokenizer; refused at load if its ids differ

                Unspecified sampling settings use the model's recommendations, then engine defaults.
                """);
    }

    static void conversationHelp(PrintStream out) {
        out.println(
                """
                Conversation options (before or after the command):
                  --system-prompt <text>      conversation instructions
                  --think inline              send thoughts to stdout instead of stderr
                  --stream / --no-stream      stream generated text (default: on)
                  --echo / --no-echo          echo token spellings to stderr (default: off)
                  --color <auto|on|off>       terminal colors (default: auto)
                                              on forces colors even with NO_COLOR or TERM=dumb
                """);
    }
}
