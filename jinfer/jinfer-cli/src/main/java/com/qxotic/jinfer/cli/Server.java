package com.qxotic.jinfer.cli;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.ChatEngine;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.llm.Sampling;
import com.qxotic.jinfer.server.ServerConfig;
import com.qxotic.jinfer.server.TranscriptionServer;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.net.BindException;
import java.net.InetSocketAddress;
import java.time.Duration;
import java.util.LinkedHashSet;
import java.util.Set;

/** HTTP application. The model selects the supported API; transport stays in jinfer-server. */
final class Server {
    static final int DEFAULT_PORT = 54154;

    private Server() {}

    static final class Settings {
        String host = "127.0.0.1";
        int port = DEFAULT_PORT;
        String apiKey;
        final Set<String> origins = new LinkedHashSet<>();
        int concurrency = ServerConfig.Limits.DEFAULTS.threads();
        Integer queueDepth;
        long maxBodyBytes = ServerConfig.Limits.DEFAULTS.maxBodyBytes();
        Duration writeTimeout = ServerConfig.Limits.DEFAULTS.writeTimeout();
        Duration requestTimeout = ServerConfig.Limits.DEFAULTS.requestTimeout();
        boolean noGrammar;

        ServerConfig.Limits limits() {
            return new ServerConfig.Limits(
                    concurrency,
                    queueDepth == null ? ServerConfig.Limits.DEFAULTS.queueCapacity() : queueDepth,
                    maxBodyBytes,
                    !noGrammar,
                    writeTimeout,
                    requestTimeout,
                    ServerConfig.Limits.DEFAULTS.shutdownTimeout());
        }
    }

    static boolean read(Options o, Options.Args a) {
        Settings s = o.server;
        switch (a.name) {
            case "--host" -> s.host = a.value();
            case "--port" -> s.port = a.integer();
            case "--api-key" -> s.apiKey = a.value();
            case "--cors-origin" -> s.origins.add(a.value());
            case "--concurrency" -> s.concurrency = a.integer();
            case "--queue-depth" -> s.queueDepth = a.integer();
            case "--max-body-mb" -> s.maxBodyBytes = (long) a.integer() << 20;
            case "--write-timeout" -> s.writeTimeout = seconds(a);
            case "--request-timeout" -> s.requestTimeout = seconds(a);
            case "--no-grammar" -> s.noGrammar = a.flag();
            default -> {
                return false;
            }
        }
        o.use(a.name, Set.of("server"));
        return true;
    }

    private static Duration seconds(Options.Args a) {
        long value = a.longValue();
        Options.require(value >= 0, "%s must be non-negative; got %s", a.name, value);
        return Duration.ofSeconds(value);
    }

    static void validate(Options o) {
        Options.require(o.input == null, "server takes no input argument");
        Settings s = o.server;
        Options.require(
                s.port >= 0 && s.port <= 65535, "--port must be within [0, 65535]; got %s", s.port);
        Options.require(!s.host.isBlank(), "--host must not be blank; got '%s'", s.host);
        Options.require(
                s.concurrency > 0, "--concurrency must be at least 1; got %s", s.concurrency);
        // The language server allocates 2*N admission permits using an int.
        Options.require(
                s.concurrency <= Integer.MAX_VALUE / 2,
                "--concurrency must not exceed %d; got %s",
                Integer.MAX_VALUE / 2,
                s.concurrency);
        Options.require(
                s.queueDepth == null || s.queueDepth >= 0,
                "--queue-depth must be non-negative; got %s",
                s.queueDepth);
        Options.require(
                s.maxBodyBytes > 0, "--max-body-mb must be positive; got %s", s.maxBodyBytes >> 20);
        Options.require(
                !s.writeTimeout.isZero(),
                "--write-timeout must be positive; got %s",
                s.writeTimeout.toSeconds());
        Options.require(
                !o.thinkInline, "server cannot route thoughts inline; use --think on or off");
    }

    static void validateTranscription(Options o) {
        o.rejectLanguageOptions("a transcription server");
        Options.require(!o.rawPrompt, "--raw-prompt does not apply to a transcription server");
        Options.require(
                o.server.queueDepth == null,
                "--queue-depth does not apply to a transcription server");
        Options.require(
                !o.server.noGrammar, "--no-grammar does not apply to a transcription server");
        Options.require(
                o.promptCache == null, "--cache/--cache-ro do not apply to a transcription server");
    }

    /** DNS and binding policy are execution work, not argument parsing. */
    static ServerConfig config(Options o, Sampling sampling) {
        Settings s = o.server;
        var address = new InetSocketAddress(s.host, s.port);
        Options.require(!address.isUnresolved(), "--host %s does not resolve", s.host);
        Options.require(
                address.getAddress().isLoopbackAddress()
                        || (s.apiKey != null && !s.apiKey.isBlank()),
                "a non-loopback --host requires --api-key");
        return new ServerConfig(
                address,
                defaults(o, sampling),
                s.limits(),
                new ServerConfig.Access(s.apiKey, s.origins.isEmpty() ? Set.of("*") : s.origins));
    }

    private static ServerConfig.Defaults defaults(Options o, Sampling sampling) {
        return new ServerConfig.Defaults(
                sampling,
                o.maxOutputTokens,
                o.think,
                o.rawPrompt,
                o.maxReasoningTokens,
                o.reasoningCutoffMessage);
    }

    static int run(Options options, Main.IO io, ModelStore store) throws IOException {
        var config =
                config(options, null); // refuse an invalid bind before fetching/loading a model
        Options.Files files = options.resolve(store);
        Arena arena = Arenas.newCrossThread();
        try {
            ChatEngine engine;
            try {
                engine = Main.openText(options, files, arena, io);
            } catch (UnsupportedOperationException notLanguage) {
                // The existing provider API reports unsupported kinds this way; isolate that
                // unsupported-kind detail here rather than making users select a task themselves.
                if (notLanguage.getMessage() == null
                        || !notLanguage.getMessage().contains("not a language architecture"))
                    throw notLanguage;
                validateTranscription(options);
                return runTranscription(files, arena, io, config);
            }
            try (engine) {
                var sampling = options.sampling(engine.loaded().samplingDefaults());
                var running =
                        startLanguage(engine, config.withDefaults(defaults(options, sampling)), io);
                return await(
                        running::await,
                        () -> {
                            running.close();
                            engine.savePrompts();
                        });
            }
        } catch (UnsupportedOperationException e) {
            throw Main.failure("model '" + files.model() + "' cannot be served", e);
        } finally {
            Arenas.close(arena);
        }
    }

    static com.qxotic.jinfer.server.Server.Running startLanguage(
            ChatEngine engine, ServerConfig config, Main.IO io) throws IOException {
        com.qxotic.jinfer.server.Server.Running running;
        try {
            running = com.qxotic.jinfer.server.Server.start(engine, config);
        } catch (BindException e) {
            throw bindFailure(config, e);
        }
        io.err()
                .printf(
                        "model       %s (context %d)%n",
                        engine.modelName(), engine.contextCapacity());
        listening(io.err(), running.address(), "OpenAI-compatible");
        return running;
    }

    private static int runTranscription(
            Options.Files files, Arena arena, Main.IO io, ServerConfig config) throws IOException {
        TranscriptionServer.Running running;
        try (var spinner = LoadSpinner.start("Loading model", io)) {
            var model = Models.loadTranscription(files.model(), arena, files.companions());
            running =
                    TranscriptionServer.start(
                            model, files.model().getFileName().toString(), config);
        } catch (BindException e) {
            throw bindFailure(config, e);
        } catch (IOException | IllegalArgumentException e) {
            throw Main.failure("cannot prepare transcription model '" + files.model() + "'", e);
        }
        listening(io.err(), running.address(), "POST /v1/audio/transcriptions");
        return await(running::await, running::close);
    }

    private static IOException bindFailure(ServerConfig config, BindException cause) {
        return new IOException(
                "port "
                        + config.bind().getPort()
                        + " on "
                        + config.bind().getHostString()
                        + " is already in use; choose another with --port",
                cause);
    }

    private static void listening(PrintStream out, InetSocketAddress address, String api) {
        String host = address.getHostString();
        if (host.contains(":")) host = "[" + host + "]";
        out.printf("listening   http://%s:%d (%s)%n", host, address.getPort(), api);
    }

    @FunctionalInterface
    interface Await {
        void run() throws InterruptedException;
    }

    /** Both transports have the same process lifetime; shutdown and normal return may race. */
    static int await(Await wait, Runnable close) {
        class Stop implements Runnable {
            boolean done;

            public synchronized void run() {
                if (done) return;
                done = true;
                close.run();
            }
        }
        Stop stop = new Stop();
        Thread hook = new Thread(stop, "jinfer-server-shutdown");
        Runtime.getRuntime().addShutdownHook(hook);
        try {
            wait.run();
            return 0;
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return 130;
        } finally {
            try {
                stop.run();
            } finally {
                try {
                    Runtime.getRuntime().removeShutdownHook(hook);
                } catch (IllegalStateException shuttingDown) {
                    /* The hook performs the same stop. */
                }
            }
        }
    }

    static void printHelp(PrintStream out) {
        out.println(
                """
                jinfer server - serve a model over HTTP (alias: serve)
                Usage: jinfer [model options] server [options]
                Examples:
                  jinfer server -m model.gguf --port 8080
                  jinfer -m parakeet.gguf server

                The model selects the language or transcription API automatically.
                Server options (after the command):
                  --host <host>              bind address; default 127.0.0.1
                  --port <int>               default 54154; 0 selects an available port
                  --api-key <token>          required for non-loopback binds
                  --cors-origin <origin>     repeatable; default *
                  --concurrency <int>        default 16; language: admits up to 2*N HTTP handlers;
                                             transcription: N handler threads, not parallel generations
                  --queue-depth <int>        waiting language generations; default 4; 0: no waiting
                                             not applicable to transcription models
                  --max-body-mb <int>        request-body limit; default 32
                  --write-timeout <seconds>  body-read/SSE-write timeout; default 30
                  --request-timeout <seconds> generation deadline; default 300; 0 disables
                  --no-grammar               disable constrained generation (language models)
                  --cache / --cache-ro <file> persistent prompt cache (language models)
                  --raw-prompt               default to raw language prompts

                Generation settings are request defaults, not HTTP resource limits.
                Clients supply system messages in their requests; --system-prompt is not a server option.
                """);
        Options.modelHelp(out);
        Options.generationHelp(out);
    }
}
