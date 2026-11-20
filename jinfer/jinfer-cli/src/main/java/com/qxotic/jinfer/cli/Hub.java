package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.cache.FrozenBlocks;
import com.qxotic.jinfer.hub.ModelRef;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import java.util.Set;
import java.util.function.UnaryOperator;

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
            case "list", "prune" ->
                    Options.require(
                            options.operands.isEmpty(), options.command + " takes no arguments");
            case "pull", "find", "remove" ->
                    Options.require(
                            !options.operands.isEmpty()
                                    && options.operands.stream().noneMatch(String::isBlank),
                            options.command + " needs at least one model reference");
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
            case "prune" -> prune(store, io.out());
            case "find" -> {
                return find(options.operands, store, io.out(), io.err());
            }
            case "remove" -> {
                return remove(options.operands, store, io.out(), io.err());
            }
            case "cache-info" -> cacheInfo(Path.of(options.operands.getFirst()), io.out());
            default -> throw new AssertionError(options.command);
        }
        return 0;
    }

    /**
     * The file {@code -m} would load for each argument, from the cache alone: no request, no
     * hashing. A miss is reported and the rest go on.
     */
    static int find(List<String> refs, ModelStore store, PrintStream out, PrintStream err) {
        int status = 0;
        for (String ref : refs) {
            try {
                Optional<Path> file = store.find(ref);
                if (file.isPresent()) {
                    out.println(file.get());
                    continue;
                }
                err.println("jinfer find: " + miss(store, ref));
            } catch (IllegalArgumentException | UncheckedIOException refused) {
                err.println("jinfer find: " + Options.rootMessage(refused));
            }
            status = 1;
        }
        return status;
    }

    /**
     * Deletes, for each argument, the one file {@code find} prints for it, and says which. A
     * repository named alone while it holds several cached files is refused with their names:
     * deleting is never a guess. A miss is reported and the rest go on.
     */
    static int remove(List<String> refs, ModelStore store, PrintStream out, PrintStream err) {
        int status = 0;
        for (String ref : refs) {
            try {
                Optional<Path> file = store.find(ref);
                List<String> choices = isRepository(ref) ? cachedFrom(store, ref) : List.of();
                if (choices.size() > 1) {
                    err.println(
                            "jinfer remove: "
                                    + ref
                                    + " holds "
                                    + choices.size()
                                    + " cached files; name the one to remove:"
                                    + menu(choices));
                } else if (file.isEmpty()) {
                    err.println("jinfer remove: " + miss(store, ref));
                } else {
                    long bytes = Files.size(file.get());
                    if (store.remove(ref)) {
                        out.println(
                                "removed "
                                        + file.get().getFileName()
                                        + " ("
                                        + humanBytes(bytes)
                                        + ")");
                        continue;
                    }
                    err.println("jinfer remove: " + ref + " is not in the model cache");
                }
            } catch (IOException | IllegalArgumentException | UncheckedIOException refused) {
                err.println("jinfer remove: " + Options.rootMessage(refused));
            }
            status = 1;
        }
        return status;
    }

    /** Why nothing was found, naming what the repository does have cached, if anything. */
    private static String miss(ModelStore store, String ref) {
        if (!ModelStore.isRef(ref)) return Options.noSuchModel(ref);
        List<String> cached = cachedFrom(store, ref);
        return "not cached: "
                + ref
                + (cached.isEmpty() ? "" : "; cached from its repository:" + menu(cached));
    }

    /** {@code owner/repo} with neither a path nor a quant. */
    private static boolean isRepository(String ref) {
        if (!ModelStore.isRef(ref)) return false;
        ModelRef parsed = ModelRef.parse(ref);
        return parsed.path().isEmpty() && parsed.quant() == null;
    }

    /** The cached models of {@code ref}'s repository, by the names {@code list} prints. */
    private static List<String> cachedFrom(ModelStore store, String ref) {
        if (!ModelStore.isRef(ref)) return List.of();
        ModelRef repo = ModelRef.parse(ref);
        return store.cached().stream()
                .map(ModelStore.Cached::ref)
                .filter(c -> isIn(c, repo))
                .map(c -> name(store, c))
                .sorted(String.CASE_INSENSITIVE_ORDER)
                .toList();
    }

    private static boolean isIn(String ref, ModelRef repo) {
        if (!ModelStore.isRef(ref)) return false;
        ModelRef file = ModelRef.parse(ref);
        return file.host().equals(repo.host()) && file.repoId().equalsIgnoreCase(repo.repoId());
    }

    private static String menu(List<String> names) {
        return "\n  " + String.join("\n  ", names);
    }

    /**
     * The shortest name that loads the same file: {@code owner/repo:QUANT} when that quant picks
     * exactly this file, else the ref without the default host. Every candidate is checked with
     * {@link ModelStore#find}, so a printed name always loads the file it stands for.
     */
    static String name(ModelStore store, String ref) {
        if (!ModelStore.isRef(ref)) return ref;
        ModelRef parsed = ModelRef.parse(ref);
        String repo =
                (parsed.host().equals("hf.co") ? "" : parsed.host() + "/")
                        + parsed.repoId()
                        + (parsed.revision() == null ? "" : "@" + parsed.revision());
        String full = repo + "/" + parsed.path();
        Optional<Path> file = store.find(ref);
        if (file.isEmpty() || parsed.path().contains("/")) return full;
        // first what follows the model's own name: LFM2.5-1.2B-Instruct-GGUF holds
        // LFM2.5-1.2B-Instruct-QAD-Q4_0, whose quant is QAD-Q4_0, though Q4_0 may load it too;
        // then each tail after a separator, shortest first: Q8_0, Q4_K_M
        String stem = parsed.path().replaceFirst("(?i)\\.gguf$", "");
        String model = parsed.repo().replaceFirst("(?i)[-_.]gguf$", "") + "-";
        List<String> quants = new ArrayList<>();
        if (stem.regionMatches(true, 0, model, 0, model.length())) {
            quants.add(stem.substring(model.length()));
        }
        for (int at = stem.length() - 1; at > 0; at--) {
            if ("-._".indexOf(stem.charAt(at - 1)) >= 0) quants.add(stem.substring(at));
        }
        for (String quant : quants) {
            // a quant reads like one (Q8_0, not 0 or M) and picks this very file; written in
            // capitals
            // whatever the file's case (kokoro-82m-q8_0.gguf is :Q8_0), as quants are, and as the
            // case-blind lookup takes it
            String tag = quant.toUpperCase(Locale.ROOT);
            if (!quant.isEmpty()
                    && Character.isLetter(quant.charAt(0))
                    && quant.chars().anyMatch(Character::isDigit)
                    && file.equals(store.find(repo + ":" + tag))) {
                return repo + ":" + tag;
            }
        }
        return full;
    }

    static void pull(List<String> refs, boolean force, ModelStore store, PrintStream out)
            throws IOException {
        Options.pullFiles(store, refs, force).forEach(out::println);
    }

    static void list(ModelStore store, PrintStream out) {
        list(store.cached(), ref -> name(store, ref), store.root(), out);
        long older = store.detached().stream().mapToLong(ModelStore.Cached::sizeBytes).sum();
        if (older > 0) {
            out.printf("%10s  in older versions: 'jinfer prune' frees it%n", humanBytes(older));
        }
    }

    /** Deletes the files of versions no branch names, saying what each one frees. */
    static void prune(ModelStore store, PrintStream out) {
        long freed = 0;
        for (var entry : store.detached()) {
            if (!store.remove(entry.ref()) || entry.sizeBytes() == 0) continue;
            ModelRef ref = ModelRef.parse(entry.ref());
            out.printf(
                    "removed %s (%s), an older version from %s%n",
                    Path.of(ref.path()).getFileName(), humanBytes(entry.sizeBytes()), ref.repoId());
            freed += entry.sizeBytes();
        }
        out.println(freed == 0 ? "no older versions cached" : "freed " + humanBytes(freed));
    }

    /**
     * The cached GGUFs {@code --model} or {@code --with} can name, by {@code name}, alphabetically.
     */
    static void list(
            List<ModelStore.Cached> cached,
            UnaryOperator<String> name,
            Path root,
            PrintStream out) {
        List<ModelStore.Cached> models =
                cached.stream()
                        .map(c -> new ModelStore.Cached(name.apply(c.ref()), c.sizeBytes()))
                        .sorted(
                                Comparator.comparing(
                                        ModelStore.Cached::ref, String.CASE_INSENSITIVE_ORDER))
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

                              -f, --force  download again even when already cached

                            Prints local paths. A failed refresh preserves the previous cached file.
                            Mutable references require network access, even when cached;
                            'jinfer find' prints a cached path without it.
                            """;
                    case "list" ->
                            """
                            jinfer list - show cached model references and sizes
                            Usage: jinfer list
                            """;
                    case "find" ->
                            """
                            jinfer find - print the path of a cached model
                            Usage: jinfer find <ref>...
                            Example: llama-server -m "$(jinfer find LiquidAI/LFM2.5-350M-GGUF:Q8_0)"

                            Never touches the network: a model that is not cached fails.
                            """;
                    case "remove" ->
                            """
                            jinfer remove - delete cached models
                            Usage: jinfer remove <ref>...
                            Example: jinfer remove LiquidAI/LFM2.5-350M-GGUF:Q8_0

                            Deletes the one file 'jinfer find' prints for each reference. A
                            repository holding several cached files asks for one of their names.
                            Files outside the cache are never deleted.

                            The Hugging Face cache ($HF_HUB_CACHE, $HF_HOME/hub or
                            ~/.cache/huggingface/hub) is shared: a model removed from it is gone
                            for every tool that reads it, whatever JINFER_MODELS says.
                            """;
                    case "prune" ->
                            """
                            jinfer prune - delete model versions no branch points to
                            Usage: jinfer prune

                            A pull that updates a model keeps the version it replaced in the
                            Hugging Face cache; prune deletes those. A model pinned to a commit
                            (owner/repo@<commit>) is one too, and downloads again when next used.
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
