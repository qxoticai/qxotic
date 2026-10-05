package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/** Downloaded model files and prompt-cache inspection, using their existing storage APIs. */
final class Hub {
    private Hub() {}

    static boolean read(Options options, Options.Args args) {
        if (!args.name.equals("--force") && !args.name.equals("-f")) return false;
        options.force = args.flag();
        options.use(args.name, Set.of("pull"));
        return true;
    }

    static void validate(Options options) {
        switch (options.command) {
            case "pull" ->
                    Options.require(
                            !options.operands.isEmpty()
                                    && options.operands.stream().noneMatch(String::isBlank),
                            "pull needs at least one model reference");
            case "list" -> Options.require(options.operands.isEmpty(), "list takes no arguments");
            case "cache-info" ->
                    Options.require(
                            options.operands.size() == 1, "cache-info takes one <file.jkv>");
            default -> throw new AssertionError(options.command);
        }
    }

    static int run(Options options, Main.IO io, ModelStore store) throws IOException {
        switch (options.command) {
            case "pull" -> pull(options.operands, options.force, store, io.out());
            case "list" -> list(store, io.out());
            case "cache-info" -> cacheInfo(Path.of(options.operands.getFirst()), io.out());
            default -> throw new AssertionError(options.command);
        }
        return 0;
    }

    static void pull(List<String> refs, boolean force, ModelStore store, PrintStream out)
            throws IOException {
        Options.pullFiles(store, refs, force).forEach(out::println);
    }

    static void list(ModelStore store, PrintStream out) {
        list(store.cached(), store.root(), out);
    }

    /**
     * The cached files {@code --model} or {@code --with} can name: GGUFs only. The cache directory
     * is shared with whatever else wrote there (ONNX exports, tokenizer text), none of it loadable.
     */
    static void list(List<ModelStore.Cached> cached, Path root, PrintStream out) {
        List<ModelStore.Cached> models =
                cached.stream()
                        .filter(c -> c.ref().toLowerCase(Locale.ROOT).endsWith(".gguf"))
                        .toList();
        if (models.isEmpty()) {
            out.println("no models cached in " + root);
            return;
        }
        // the fixed-width column first: a name's display width is not its length (CJK, accents)
        long total = 0;
        for (var model : models) {
            total += model.sizeBytes();
            out.printf("%10s  %s%n", humanBytes(model.sizeBytes()), model.ref());
        }
        out.printf("%10s  total%n", humanBytes(total));
    }

    static void cacheInfo(Path file, PrintStream out) throws IOException {
        Options.requireFile(file);
        try {
            out.print(FrozenBlocks.describe(file));
        } catch (IOException e) {
            throw Main.failure("cannot inspect prompt cache '" + file + "'", e);
        }
    }

    private static String humanBytes(long bytes) {
        if (bytes < 1024) return bytes + " B";
        String[] units = {"KB", "MB", "GB", "TB"};
        double value = bytes;
        int unit = -1;
        while (value >= 1024 && unit < units.length - 1) {
            value /= 1024;
            unit++;
        }
        return String.format(Locale.ROOT, value >= 100 ? "%.0f %s" : "%.1f %s", value, units[unit]);
    }

    static void printHelp(String command, PrintStream out) {
        out.println(
                switch (command) {
                    case "pull" ->
                            """
                            jinfer pull - check upstream and download changed or missing files
                            Usage: jinfer pull [--force] <ref>...
                            Example: jinfer pull LiquidAI/LFM2.5-350M-GGUF:Q8_0

                              -f, --force  download again even when a completed file is cached

                            Prints local paths. A failed refresh preserves the previous cached file.
                            Mutable references require network access, even when cached.
                            """;
                    case "list" ->
                            """
                            jinfer list - show cached model references and sizes
                            Usage: jinfer list
                            """;
                    case "cache-info" ->
                            """
                            jinfer cache-info - inspect a prompt/KV cache (not the downloaded-model cache)
                            Usage: jinfer cache-info <file.jkv>
                            Example: jinfer cache-info prompts.jkv
                            """;
                    default -> throw new AssertionError(command);
                });
    }
}
