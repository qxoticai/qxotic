// jinfer: LLM inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects
// Related project: https://github.com/mukel/llama3.java
//
// Supports GGUF models and multiple tensor formats
// Matrix-vector kernels use Java's Vector API
// CLI modes: --chat and --instruct (the OpenAI server lands with jinfer-xserver)
//
// Build/run: `mvn package` then `java -jar target/xjinfer.jar --help`.
package com.qxotic.jinfer.x.cli;

import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.x.cache.PromptCache;
import com.qxotic.jinfer.x.chat.ChatEngine;
import com.qxotic.jinfer.x.chat.LoadedModel;
import com.qxotic.jinfer.x.llm.Sampling;
import java.io.BufferedOutputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.NoSuchFileException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Locale;

/**
 * The entry point, and only that: console setup, the verb dispatch ({@code pull}, {@code list}),
 * the model load, and the handoff to one of the two modes - {@link Instruct}, {@link Chat}. Flags
 * live in {@link Options}; the per-turn terminal rendering they share lives in {@link Turn}; the
 * generation machinery itself is {@link ChatEngine}'s.
 */
public class Main {

    public static void main(String[] args) throws IOException {
        forceUtf8Console();
        oneLineLogs();
        if (args.length > 0 && !args[0].startsWith("-")) {
            if (args[0].equals("list")) {
                Options.require(args.length == 1, "list takes no arguments");
                list();
                return;
            }
            if (!args[0].equals("pull")) {
                System.err.println(
                        "ERROR unknown command: " + args[0] + " (the only one is: pull)");
                System.err.println();
                Options.printUsage(System.err);
                System.exit(2);
                return;
            }
            pull(Arrays.copyOfRange(args, 1, args.length));
            return;
        }
        Options options;
        try {
            options = Options.parse(args);
        } catch (Options.ResolveFailure e) {
            System.err.println("ERROR " + Options.rootMessage(e));
            System.exit(1);
            return;
        } catch (IllegalArgumentException e) {
            System.err.println("ERROR " + e.getMessage());
            System.err.println();
            Options.printUsage(System.err);
            System.exit(1);
            return;
        }
        LoadedModel<?> model;
        try {
            // ONE load path: every file - model and companions - uses its preload when it has
            // one, parses fresh when it does not; any mix composes
            model = AOT.load(options.modelPath(), options.companions(), options.tokenizerPath());
        } catch (IllegalArgumentException
                | IllegalStateException
                | UnsupportedOperationException
                | UncheckedIOException
                | NoSuchFileException e) {
            // load errors carry their remedy in the message (wrong mmproj, unknown architecture,
            // split GGUF, bad pre-tokenizer flag, ...) - print it, don't bury it in a stack
            // trace; anything else is a bug and still traces
            System.err.println("ERROR " + e.getMessage());
            System.exit(1);
            return;
        }
        try {
            options.requireFitsModel(model); // --context-capacity against the model's own
        } catch (IllegalArgumentException e) {
            System.err.println("ERROR " + e.getMessage());
            System.exit(1);
            return;
        }
        Sampling sampling = options.sampling(model.samplingDefaults());
        // the engine owns the prompt cache: instruct's --cache file rides the catalog options,
        // chat gets the in-memory defaults (its own flag validation already refused --cache)
        PromptCache.Options cacheOptions =
                PromptCache.Options.DEFAULTS
                        .withContextCapacity(options.contextCapacity())
                        .withCatalog(options.promptCache(), options.promptCacheReadOnly());
        ChatEngine engine =
                new ChatEngine(model, options.modelPath().getFileName().toString(), cacheOptions)
                        .speculationDepth(options.speculationDepth());
        try {
            if (options.server()) {
                Serve.run(engine, model, sampling, options);
            } else if (options.interactive()) {
                Chat.run(engine, sampling, options);
            } else {
                Instruct.run(engine, sampling, options);
            }
        } finally {
            engine.close();
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
            System.err.println("  xjinfer pull hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M");
            System.err.println("  xjinfer pull --force <ref>    re-download even if cached");
            System.exit(2);
            return;
        }
        try {
            if (force) {
                refs.forEach(ModelStore::evict);
            }
            // several refs download concurrently; paths print in argument order, so it pipes
            ModelStore.resolveAll(refs).forEach(System.out::println);
        } catch (RuntimeException e) {
            System.err.println("ERROR " + Options.rootMessage(e));
            System.exit(1);
        }
    }

    /**
     * What the local cache holds, as refs with their sizes. Every line pastes straight back into
     * {@code --model} or {@code --with}, which is the point: the cache path IS the ref.
     */
    private static void list() {
        List<ModelStore.Cached> models = ModelStore.cached();
        if (models.isEmpty()) {
            System.out.println("no models cached in " + ModelStore.root());
            return;
        }
        int width = models.stream().mapToInt(m -> m.ref().length()).max().orElse(0);
        long total = 0;
        for (ModelStore.Cached model : models) {
            total += model.sizeBytes();
            System.out.printf(
                    "%-" + width + "s  %10s%n", model.ref(), humanBytes(model.sizeBytes()));
        }
        System.out.printf("%-" + width + "s  %10s%n", "total", humanBytes(total));
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

    /**
     * Force UTF-8 on the console so multilingual model output/input isn't garbled by a legacy code
     * page (Windows defaults stdout/stdin to one; Linux/macOS are already UTF-8, so this is a no-op
     * re-wrap). Raw-byte writes - the streamed token bytes - pass through unchanged; only String
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
}
