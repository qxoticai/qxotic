package com.qxotic.jinfer.server;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpHandler;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Semaphore;
import java.util.function.Supplier;

/**
 * The transport a single-task server (transcription, retrieval) shares, on the language server's
 * rules: an open {@code /health}, keyed {@code /props}, {@code /metrics} and {@code /v1/models},
 * work routes admitted through the gate ({@code --concurrency} at once, then 503 + Retry-After), a
 * JSON 404 for every other path, and the access log on all of them. A task server registers its
 * work routes and starts; what differs between them is only the work.
 */
final class TaskTransport {

    private final ServerConfig config;
    private final HttpServer http;
    private final Semaphore admissions;
    private final long startNanos = System.nanoTime();

    TaskTransport(ServerConfig config) throws IOException {
        this.config = config;
        this.http = HttpServer.create(config.bind(), 0);
        this.admissions = new Semaphore(config.limits().threads());
    }

    /** Handlers admitted through the gate right now. */
    int inFlight() {
        return config.limits().threads() - admissions.availablePermits();
    }

    double uptimeSeconds() {
        return (System.nanoTime() - startNanos) / 1e9;
    }

    /**
     * The probes. Never gated: a probe runs no model work, and a load balancer or a scrape needs it
     * most in exactly the saturated state the gate reports. {@code /health} carries no key, as the
     * language server's does; {@code props} and {@code metrics} are built per request.
     */
    void probes(
            String servedModel,
            Map<String, Object> modelCard,
            Supplier<Map<String, Object>> props,
            Supplier<String> metrics) {
        ServerConfig.Access probe = new ServerConfig.Access(null, config.access().allowedOrigins());
        probe(
                "/health",
                probe,
                exchange ->
                        Http.sendJson(
                                exchange,
                                200,
                                JsonCodec.object(
                                        "status", "ok", "busy", inFlight() > 0, "queued", 0)));
        probe("/props", config.access(), exchange -> Http.sendJson(exchange, 200, props.get()));
        probe(
                "/metrics",
                config.access(),
                exchange -> Http.sendText(exchange, 200, Metrics.CONTENT_TYPE, metrics.get()));
        context(
                "/v1/models",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    if (Http.requireMethod(exchange, "GET")) return;
                    String path = exchange.getRequestURI().getPath();
                    if (path.equals("/v1/models")) {
                        Http.sendJson(
                                exchange,
                                200,
                                JsonCodec.object("object", "list", "data", List.of(modelCard)));
                    } else if (path.equals("/v1/models/" + servedModel)) {
                        Http.sendJson(exchange, 200, modelCard);
                    } else if (path.startsWith("/v1/models/")) {
                        Http.sendError(
                                exchange,
                                404,
                                "Unknown model: "
                                        + path.substring("/v1/models/".length())
                                        + " (this server serves "
                                        + servedModel
                                        + ")");
                    } else {
                        Http.sendError(exchange, 404, "Not found"); // /v1/modelsXYZ: a wrong path
                    }
                });
    }

    private void probe(String path, ServerConfig.Access access, HttpHandler answer) {
        context(
                path,
                exchange -> {
                    if (Http.preamble(exchange, access)) return;
                    // contexts match by PREFIX: /healthXYZ is a wrong path, not the probe
                    if (!path.equals(exchange.getRequestURI().getPath())) {
                        Http.sendError(exchange, 404, "Not found");
                        return;
                    }
                    if (Http.requireMethod(exchange, "GET")) return;
                    answer.handle(exchange);
                });
    }

    /** One request's work, after the transport has admitted it and checked path and method. */
    interface Work {
        void handle(HttpExchange exchange) throws IOException;
    }

    /**
     * A POST endpoint doing model work, admitted through the gate. The language server's rule for
     * failures: only a validator's two types are the client's fault - a 400 naming the field, or a
     * 404 for a model this server does not serve - and anything else is ours, logged, not echoed.
     */
    void work(String path, Work work) {
        context(
                path,
                Server.gated(
                        exchange -> {
                            if (Http.preamble(exchange, config.access())) return;
                            if (!path.equals(exchange.getRequestURI().getPath())) {
                                Http.sendError(exchange, 404, "Not found");
                                return;
                            }
                            if (Http.requireMethod(exchange, "POST")) return;
                            try {
                                work.handle(exchange);
                            } catch (IllegalArgumentException | UnsupportedOperationException e) {
                                Http.sendErrorQuietly(
                                        exchange,
                                        Server.clientStatus(e),
                                        Http.errorMessage(e),
                                        Values.param(e));
                            } catch (RuntimeException e) {
                                Log.LOG.log(
                                        System.Logger.Level.ERROR,
                                        "unhandled fault serving " + path,
                                        e);
                                Http.sendErrorQuietly(exchange, 500, "Internal server error");
                            }
                        },
                        admissions,
                        config.limits()));
    }

    /**
     * Registers the catch-all and starts serving. Every other path gets the JSON 404 the language
     * server answers, not the JDK's HTML page, saying what this server does serve; {@code onClose}
     * runs once the transport has stopped, after the last handler.
     */
    Running start(String serves, String threadName, Runnable onClose) {
        context(
                "/",
                exchange -> {
                    if (Http.preamble(exchange, config.access())) return;
                    Http.sendError(
                            exchange,
                            404,
                            "unknown path "
                                    + exchange.getRequestURI().getPath()
                                    + "; this server serves "
                                    + serves);
                });
        // every exchange gets a thread at once; the gate bounds the work, not the probes
        http.setExecutor(
                Executors.newCachedThreadPool(
                        runnable -> {
                            Thread thread = new Thread(runnable, threadName);
                            thread.setDaemon(true);
                            return thread;
                        }));
        http.start();
        long stopDelay = Math.ceilDiv(config.limits().shutdownTimeout().toNanos(), 1_000_000_000L);
        return new Running(http, (int) Math.min(Integer.MAX_VALUE, stopDelay), onClose);
    }

    private void context(String path, HttpHandler handler) {
        Http.logged(http.createContext(path, handler));
    }

    /** The started transport: what each task server's public {@code Running} delegates to. */
    static final class Running {
        private final HttpServer http;
        private final int stopDelaySeconds;
        private final Runnable onClose;
        private final CountDownLatch stopped = new CountDownLatch(1);
        private boolean closed;

        private Running(HttpServer http, int stopDelaySeconds, Runnable onClose) {
            this.http = http;
            this.stopDelaySeconds = stopDelaySeconds;
            this.onClose = onClose;
        }

        InetSocketAddress address() {
            return http.getAddress();
        }

        void await() throws InterruptedException {
            stopped.await();
        }

        synchronized void close() {
            if (closed) return;
            closed = true;
            try {
                http.stop(stopDelaySeconds);
                if (http.getExecutor() instanceof ExecutorService pool) pool.shutdownNow();
                onClose.run();
            } finally {
                stopped.countDown();
            }
        }
    }
}
