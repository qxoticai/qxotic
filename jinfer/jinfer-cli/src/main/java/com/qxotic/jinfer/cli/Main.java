// jinfer: inference in pure Java
// Author: Alfonso² Peterssen
// Based on Andrej Karpathy's llama2.c and minbpe projects.
// Related project: https://github.com/mukel/llama3.java
package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.ChatEngine;
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
            Options.require(text != null && !text.isBlank(), "Input requires non-blank text");
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
        } catch (Options.Usage e) {
            String command = options == null ? e.command : options.command;
            String name = "jinfer" + (command == null ? "" : " " + command);
            io.err().println(name + ": " + e.getMessage());
            if (e.showHelp)
                io.err().println("Run '" + name + " --help' for available commands and options.");
            return 2;
        } catch (IOException | UncheckedIOException e) {
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

    /** A useful headline first, backend details below it; retain the cause for diagnostics. */
    static IOException failure(String summary, Throwable cause) {
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
        } catch (UnsupportedOperationException e) {
            throw failure(
                    "cannot run " + options.command + " with model '" + files.model() + "'", e);
        } finally {
            Arenas.close(arena);
        }
    }

    static ChatEngine openText(Options options, Options.Files files, Arena arena, IO io)
            throws IOException {
        try (var spinner = LoadSpinner.start("Loading model", io)) {
            var model = AOT.load(files.model(), files.companions(), files.tokenizer(), arena);
            try {
                return new ChatEngine(
                                model,
                                files.model().getFileName().toString(),
                                options.cacheOptions())
                        .speculationDepth(options.speculationDepth);
            } catch (IllegalArgumentException | UncheckedIOException e) {
                throw failure("cannot initialize model state for '" + files.model() + "'", e);
            }
        }
    }

    private static PrintStream utf8Stream(FileDescriptor fd) {
        return new PrintStream(
                new BufferedOutputStream(new FileOutputStream(fd), 8192),
                true,
                StandardCharsets.UTF_8);
    }
}
