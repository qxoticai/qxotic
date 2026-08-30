package com.qxotic.jinfer.server;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Server-Sent-Events transport: the per-response {@link Stream} (frame encoding, flush, and the
 * checked→unchecked bridge so sinks deep in the generation loop just call {@link Stream#emit}),
 * plus a watchdog that closes a streaming client whose write has stalled - the JDK server has no
 * write timeout, so without it one dead client would wedge the single generation worker forever.
 */
final class Sse {

    private Sse() {}

    /** Every in-flight stream, watched by the stall reaper. */
    private static final Set<Stream> ACTIVE = ConcurrentHashMap.newKeySet();

    /**
     * Opens an SSE response: sets the event-stream headers, registers it with the reaper, and wraps
     * the body so each write is timed.
     */
    static Stream begin(HttpExchange exchange, Duration writeTimeout) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0);
        Stream stream = new Stream(exchange, writeTimeout.toNanos());
        ACTIVE.add(stream);
        return stream;
    }

    /**
     * Runs an SSE body, turning failures into a clean stream close instead of a hung client: a lost
     * connection ({@link UncheckedIOException} from a frame write) propagates as IOException for
     * the handler to log; any other error is delivered as a terminal in-band error event + [DONE]
     * so the client stops.
     */
    static void guarded(Stream sse, Metrics metrics, Runnable body) throws IOException {
        guarded(sse, metrics, body, false);
    }

    static void guardedResponses(Stream sse, Metrics metrics, Runnable body) throws IOException {
        guarded(sse, metrics, body, true);
    }

    private static void guarded(Stream sse, Metrics metrics, Runnable body, boolean responses)
            throws IOException {
        try {
            body.run();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        } catch (IllegalArgumentException | UnsupportedOperationException e) {
            metrics.record(Metrics.Outcome.INVALID_REQUEST);
            terminal(sse, 400, Http.errorMessage(e), responses);
        } catch (RuntimeException e) {
            metrics.record(Metrics.Outcome.FAILED);
            Log.LOG.log(System.Logger.Level.ERROR, "streaming request failed", e);
            terminal(sse, 500, "Internal server error", responses);
        }
    }

    private static void terminal(Stream sse, int status, String message, boolean responses)
            throws IOException {
        try {
            if (responses) {
                Map<String, Object> error = new LinkedHashMap<>();
                error.put("type", "error");
                error.put("code", status >= 500 ? "server_error" : "invalid_request_error");
                error.put("message", message);
                error.put("param", null);
                sse.emit("error", error);
            } else {
                sse.emit(Map.of("error", Http.errorPayload(status, message)));
            }
            sse.done();
        } catch (UncheckedIOException disconnected) {
            throw disconnected.getCause();
        }
    }

    /**
     * A reaper closes any stream whose in-flight write has blocked past ITS OWN server's write
     * timeout; the blocked write then fails with IOException, aborting that generation cleanly. One
     * per server, owned and stopped by its {@code Running}; the threshold rides on each stream, so
     * two servers with different timeouts each reap their own streams correctly.
     */
    static Thread startReaper() {
        return Thread.ofPlatform()
                .name("sse-write-reaper")
                .daemon(true)
                .start(
                        () -> {
                            while (!Thread.currentThread().isInterrupted()) {
                                try {
                                    Thread.sleep(1_000);
                                } catch (InterruptedException e) {
                                    return;
                                }
                                long now = System.nanoTime();
                                for (Stream stream : ACTIVE) {
                                    long start = stream.writeStartNanos;
                                    if (start != 0 && now - start > stream.writeStallNanos) {
                                        Log.LOG.log(
                                                System.Logger.Level.WARNING,
                                                () ->
                                                        "closing stalled streaming client "
                                                                + stream.exchange
                                                                        .getRemoteAddress());
                                        ACTIVE.remove(stream);
                                        stream.exchange.close();
                                    }
                                }
                            }
                        });
    }

    /**
     * A live SSE response. Owns the byte encoding, the per-frame flush, and the checked→unchecked
     * bridge so callers - including streaming sinks invoked deep in the generation loop - just call
     * {@link #emit}/{@link #done}.
     */
    static final class Stream implements AutoCloseable {
        private final HttpExchange exchange;
        private final OutputStream out;
        private final long writeStallNanos;
        private volatile long writeStartNanos; // 0 = no write in flight
        private int sequence;

        private Stream(HttpExchange exchange, long writeStallNanos) {
            this.exchange = exchange;
            this.writeStallNanos = writeStallNanos;
            this.out = exchange.getResponseBody();
        }

        /** A {@code data:} frame carrying one JSON value. */
        void emit(Object value) {
            frame("data: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** A named SSE event ({@code event:} line + {@code data:} frame) - the Responses API. */
        void emit(String event, Object value) {
            if (value instanceof Map<?, ?> map) {
                Map<Object, Object> numbered = new LinkedHashMap<>(map);
                numbered.putIfAbsent("sequence_number", sequence++);
                value = numbered;
            }
            frame("event: " + event + "\ndata: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** The terminal {@code [DONE]} sentinel. */
        void done() {
            frame("data: [DONE]\n\n");
        }

        private void frame(String text) {
            writeStartNanos = System.nanoTime();
            try {
                out.write(text.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                throw new UncheckedIOException(e); // client gone; unwound by guarded()
            } finally {
                writeStartNanos = 0;
            }
        }

        @Override
        public void close() throws IOException {
            ACTIVE.remove(this);
            out.close();
        }
    }
}
