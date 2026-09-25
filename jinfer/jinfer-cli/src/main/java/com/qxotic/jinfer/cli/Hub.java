package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
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
        try {
            if (force) refs.forEach(store::evict);
        } catch (IllegalArgumentException | UncheckedIOException e) {
            throw Main.failure("cannot refresh cached model files", e);
        }
        Options.resolveFiles(store, refs).forEach(out::println);
    }

    static void list(ModelStore store, PrintStream out) {
        list(store.cached(), store.root(), out);
    }

    static void list(List<ModelStore.Cached> models, Path root, PrintStream out) {
        if (models.isEmpty()) {
            out.println("no models cached in " + root);
            return;
        }
        int width = models.stream().mapToInt(m -> m.ref().length()).max().orElse(0);
        long total = 0;
        for (var model : models) {
            total += model.sizeBytes();
            out.printf("%-" + width + "s  %10s%n", model.ref(), humanBytes(model.sizeBytes()));
        }
        out.printf("%-" + width + "s  %10s%n", "total", humanBytes(total));
    }

    static void cacheInfo(Path file, PrintStream out) throws IOException {
        if (!Files.isRegularFile(file)) throw new IOException("no such file: " + file);
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
                            jinfer pull - download model files and print their local paths
                            Usage: jinfer pull [--force] <ref>...
                            Example: jinfer pull LiquidAI/LFM2.5-350M-GGUF:Q8_0

                              -f, --force  re-download even if cached
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
