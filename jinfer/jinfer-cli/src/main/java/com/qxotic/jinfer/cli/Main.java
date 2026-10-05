// jinfer: inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects.
// Related project: https://github.com/mukel/llama3.java
package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import java.io.BufferedOutputStream;
import java.io.FileDescriptor;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintStream;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;

/** Process setup and dispatch. All work returns through this boundary before the process exits. */
public final class Main {
    private Main() {}

    public static void main(String[] args) {
        System.setOut(utf8Stream(FileDescriptor.out));
        System.setErr(utf8Stream(FileDescriptor.err));
        String format = "java.util.logging.SimpleFormatter.format";
        if (System.getProperty(format) == null)
            System.setProperty(format, "%1$tT %4$-7s %5$s%6$s%n");
        System.exit(run(args, IO.system(), ModelStore.standard()));
    }

    /** Borrowed streams. Custom streams are plain output, never the process's native terminal. */
    record IO(InputStream in, PrintStream out, PrintStream err) {
        static IO system() {
            return new IO(System.in, System.out, System.err);
        }

        boolean isTerminal(int fd) {
            boolean system =
                    switch (fd) {
                        case 0 -> in == System.in;
                        case 1 -> out == System.out;
                        case 2 -> err == System.err;
                        default -> false;
                    };
            return system && Terminal.isTerminal(fd);
        }

        String text(String input) throws IOException {
            String text =
                    "-".equals(input) ? new String(read("text"), StandardCharsets.UTF_8) : input;
            Options.require(text != null && !text.isBlank(), "input requires non-blank text");
            return text;
        }

        byte[] read(String kind) throws IOException {
            try {
                return in.readAllBytes();
            } catch (IOException e) {
                throw failure("cannot read " + kind + " from stdin", e);
            }
        }
    }

    /**
     * The exit status of one invocation. The library refuses the user's input with the JDK's own
     * types - {@link IllegalArgumentException}, {@link IllegalStateException}, {@link
     * UnsupportedOperationException} - and those read as one line, like an I/O failure. Any other
     * runtime exception is a bug and keeps its stack trace.
     */
    static int run(String[] args, IO io, ModelStore store) {
        Options options = null;
        try {
            options = Options.parse(args);
            int status = 0;
            if (options.help) {
                options.printHelp(io.out());
            } else if (options.version) {
                String version = Main.class.getPackage().getImplementationVersion();
                io.out().println("jinfer " + (version == null ? "development" : version));
            } else if (!Options.MODEL_COMMANDS.contains(options.command)) {
                status = Hub.run(options, io, store);
            } else {
                options.configureRuntime();
                status =
                        switch (options.command) {
                            case "speak" -> Speak.run(options, io, store);
                            case "transcribe" -> Transcribe.run(options, io, store);
                            case "server" -> Server.run(options, io, store);
                            case "chat", "instruct" -> runText(options, io, store);
                            default -> throw new AssertionError(options.command);
                        };
            }
            if (status == 0 && io.out().checkError())
                throw new IOException("cannot write to stdout");
            return status;
        } catch (Options.UsageException e) {
            String command = options == null ? e.command : options.command;
            String name = "jinfer" + (command == null ? "" : " " + command);
            io.err().println(name + ": " + e.getMessage());
            if (e.showHelp)
                io.err()
                        .println(
                                "Run '"
                                        + name
                                        + " --help' for available "
                                        + (command == null ? "commands and options." : "options."));
            return 2;
        } catch (IOException
                | UncheckedIOException
                | IllegalArgumentException
                | IllegalStateException
                | UnsupportedOperationException e) {
            io.err().println(name(options) + ": " + Options.rootMessage(e));
            return Thread.currentThread().isInterrupted() ? 130 : 1;
        } catch (RuntimeException e) {
            if (Thread.currentThread().isInterrupted()) {
                io.err().println(name(options) + ": interrupted");
                return 130;
            }
            io.err().println(name(options) + ": unexpected failure: " + Options.rootMessage(e));
            e.printStackTrace(io.err());
            return 1;
        }
    }

    private static String name(Options options) {
        return "jinfer" + (options == null || options.command == null ? "" : " " + options.command);
    }

    /**
     * An I/O failure with the context the JDK's message lacks - the operation, the file, a remedy -
     * and the details below it. Only for I/O: a refusal keeps its own type (see {@link #run}).
     */
    static IOException failure(String summary, IOException cause) {
        return new IOException(
                summary + "\n  " + Options.rootMessage(cause).replace("\n", "\n  "), cause);
    }

    private static int runText(Options options, IO io, ModelStore store) throws IOException {
        // Read one-shot stdin before loading weights; an empty pipe should fail immediately.
        String text = options.command.equals("instruct") ? io.text(options.input) : null;
        Options.Files files = options.resolve(store);
        Arena arena = Arenas.newCrossThread();
        try (ChatEngine engine = openText(options, files, arena, io)) {
            var sampling = options.sampling(engine.loaded().samplingDefaults());
            if (options.command.equals("chat")) Chat.run(engine, sampling, options, io);
            else Instruct.run(engine, sampling, options, io, text);
            return Thread.currentThread().isInterrupted() ? 130 : 0;
        } catch (ModelProvider.IncompatibleModelException notLanguage) {
            throw unrunnable(options, files.model(), arena, notLanguage);
        } finally {
            Arenas.close(arena);
        }
    }

    static ChatEngine openText(Options options, Options.Files files, Arena arena, IO io)
            throws IOException {
        try (var spinner = LoadSpinner.start("Loading model", io)) {
            return loadText(options, files, arena);
        }
    }

    /** As {@link #openText} without the spinner, for a caller that shows its own. */
    static ChatEngine loadText(Options options, Options.Files files, Arena arena)
            throws IOException {
        var model = AOT.load(files.model(), files.companions(), files.tokenizer(), arena);
        ChatEngine engine =
                new ChatEngine(
                                model,
                                files.model().getFileName().toString(),
                                options.cacheOptions())
                        .speculationDepth(options.speculationDepth);
        // an explicit depth on a model that cannot draft would be accepted and do nothing; 0 is
        // "off", meaningful everywhere, and the default stays a default
        if (options.supplied("--speculation-depth")
                && options.speculationDepth > 0
                && !engine.speculationReady()) {
            engine.close();
            throw new IllegalArgumentException(
                    "--speculation-depth needs a draft head, which this model does not have;"
                            + " --with speculation=<file> attaches one where the architecture"
                            + " offers it");
        }
        return engine;
    }

    static final String MODELS_DOCS =
            "https://github.com/qxoticai/qxotic/blob/main/docs/jinfer/index.md#models-and-capabilities";

    /**
     * The refusal for a model no command here runs. An embedding or reranking checkpoint is the one
     * people reach for, and the loader's own refusal speaks to library callers, so the CLI names it
     * and where it does run; any other model keeps the loader's words.
     */
    static IllegalArgumentException unrunnable(
            Options options,
            Path model,
            Arena arena,
            ModelProvider.IncompatibleModelException refusal) {
        if (!retrieval(model, arena)) return refusal;
        return new IllegalArgumentException(
                "'"
                        + options.modelRef
                        + "' is an embedding or reranking model, which the jinfer CLI does not run"
                        + " yet; use it from Java with Models.loadEmbedder or Models.loadReranker: "
                        + MODELS_DOCS,
                refusal);
    }

    /** Whether the model's port loads it as an embedder: only an error path asks, so it loads. */
    private static boolean retrieval(Path model, Arena arena) {
        try {
            Models.loadEmbedder(model, arena);
            return true;
        } catch (ModelProvider.IncompatibleModelException notAnEmbedder) {
            return false;
        } catch (IOException | RuntimeException failedAsOne) {
            return true; // its port took it as an embedder: that is what it is
        }
    }

    private static PrintStream utf8Stream(FileDescriptor fd) {
        return new PrintStream(
                new BufferedOutputStream(new FileOutputStream(fd), 8192),
                true,
                StandardCharsets.UTF_8);
    }
}
