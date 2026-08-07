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
package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Thinking;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampler;
import com.qxotic.jinfer.llm.Sampling;
import java.io.BufferedOutputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

/**
 * The entry point, and only that: console setup, the verb dispatch ({@code pull}, {@code list}),
 * the model load, and the handoff to one of the three modes - {@link Instruct}, {@link Chat},
 * {@link Serve}. Flags live in {@link Options}; the per-turn generation they all share lives in
 * {@link Turn}.
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
            pull(java.util.Arrays.copyOfRange(args, 1, args.length));
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
            if (!options.companions().isEmpty()) {
                // a model with companions is never AOT-preloaded: load the set fresh
                model =
                        Models.load(
                                options.modelPath(),
                                java.lang.foreign.Arena.global(),
                                options.companions());
            } else {
                model = AOT.tryUsePreLoaded(options.modelPath());
                if (model == null) {
                    model = Models.load(options.modelPath(), java.lang.foreign.Arena.global());
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
        Sampling sampling = options.sampling(model.samplingDefaults());
        if (options.server()) {
            Serve.run(model, options, sampling);
            return;
        }
        try {
            options.requireFitsModel(model); // --context-capacity against the model's own
        } catch (IllegalArgumentException e) {
            System.err.println("ERROR " + e.getMessage());
            System.exit(1);
            return;
        }
        Sampler sampler = sampling.sampler(model.model().config().vocabularySize());
        if (!options.think()) {
            sampler = Thinking.banMarkers(sampler, model.tokenizer());
        }
        run(model, sampler, options);
    }

    /**
     * The mode dispatch: a one-shot {@code --prompt} or an interactive {@code --chat} loop. Models
     * exposing a {@link ChatTemplate} run through the model's own codec framing - native codecs
     * incrementally via whole-conversation re-encode + longest-common-prefix reuse (the verbatim
     * splice keeps generated turns in the common prefix); everything else falls back to the
     * whole-render Jinja path with a fresh state per turn.
     */
    private static <S extends RuntimeState> void run(
            LoadedModel<S> model, Sampler sampler, Options options) throws IOException {
        ChatTemplate template = options.rawPrompt() ? null : model.template().orElse(null);
        if (!options.interactive()) {
            Instruct.run(model, template, sampler, options);
        } else if (template != null) {
            Chat.runCodec(model, template, sampler, options);
        } else {
            Chat.runWholeRender(model, sampler, options);
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
                System.err.println("ERROR " + Options.rootMessage(e));
                System.exit(1);
                return;
            }
        }
    }

    /**
     * What the local cache holds, as refs with their sizes. Every line pastes straight back into
     * {@code --model} or {@code --with}, which is the point: the cache path IS the ref.
     */
    private static void list() {
        java.util.List<ModelStore.Cached> models = ModelStore.cached();
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
