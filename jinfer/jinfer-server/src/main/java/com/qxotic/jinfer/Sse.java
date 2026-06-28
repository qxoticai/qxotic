package com.qxotic.jinfer;

import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;

import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Server-Sent-Events transport: the per-response {@link Stream} (frame encoding, flush, and the
 * checked→unchecked bridge so sinks deep in the generation loop just call {@link Stream#emit}),
 * plus a watchdog that closes a streaming client whose write has stalled — the JDK server has no
 * write timeout, so without it one dead client would wedge the single generation worker forever.
 */
final class Sse {

    private Sse() {}

    /** Every in-flight stream, watched by the stall reaper. */
    private static final Set<Stream> ACTIVE = ConcurrentHashMap.newKeySet();

    /** Opens an SSE response: sets the event-stream headers, registers it with the reaper, and
     *  wraps the body so each write is timed. */
    static Stream begin(HttpExchange exchange) throws IOException {
        Headers headers = exchange.getResponseHeaders();
        headers.set("Content-Type", "text/event-stream; charset=utf-8");
        headers.set("Cache-Control", "no-cache");
        exchange.sendResponseHeaders(200, 0);
        Stream stream = new Stream(exchange);
        ACTIVE.add(stream);
        return stream;
    }

    /** Runs an SSE body, turning failures into a clean stream close instead of a hung client: a
     *  lost connection ({@link UncheckedIOException} from a frame write) propagates as IOException
     *  for the handler to log; any other error is delivered as a terminal in-band error event +
     *  [DONE] so the client stops. */
    static void guarded(Stream sse, Runnable body) throws IOException {
        try {
            body.run();
        } catch (UncheckedIOException e) {
            throw e.getCause();
        } catch (RuntimeException e) {
            sse.emit(Map.of("error", Http.errorPayload(400, Http.errorMessage(e))));
            sse.done();
        }
    }

    /** A reaper closes any stream whose in-flight write has blocked past
     *  {@code llama.serverWriteTimeout}; the blocked write then fails with IOException, aborting
     *  that generation cleanly. */
    static void startReaper() {
        Thread.ofPlatform().name("sse-write-reaper").daemon(true).start(() -> {
            while (true) {
                try {
                    Thread.sleep(5_000);
                } catch (InterruptedException e) {
                    return;
                }
                long now = System.nanoTime();
                for (Stream stream : ACTIVE) {
                    long start = stream.writeStartNanos;
                    if (start != 0 && now - start > RuntimeFlags.SERVER_WRITE_STALL_NANOS) {
                        System.err.println("closing stalled streaming client " + stream.exchange.getRemoteAddress());
                        ACTIVE.remove(stream);
                        stream.exchange.close();
                    }
                }
            }
        });
    }

    /** A live SSE response. Owns the byte encoding, the per-frame flush, and the checked→unchecked
     *  bridge so callers — including streaming sinks invoked deep in the generation loop — just
     *  call {@link #emit}/{@link #done}. */
    static final class Stream implements AutoCloseable {
        private final HttpExchange exchange;
        private final OutputStream out;
        private volatile long writeStartNanos; // 0 = no write in flight

        private Stream(HttpExchange exchange) {
            this.exchange = exchange;
            this.out = new FilterOutputStream(exchange.getResponseBody()) {
                @Override public void write(byte[] b, int off, int len) throws IOException {
                    writeStartNanos = System.nanoTime();
                    try {
                        super.out.write(b, off, len);
                    } finally {
                        writeStartNanos = 0;
                    }
                }
            };
        }

        /** A {@code data:} frame carrying one JSON value. */
        void emit(Object value) {
            frame("data: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** A named SSE event ({@code event:} line + {@code data:} frame) — the Responses API. */
        void emit(String event, Object value) {
            frame("event: " + event + "\ndata: " + JsonCodec.stringify(value) + "\n\n");
        }

        /** The terminal {@code [DONE]} sentinel. */
        void done() {
            frame("data: [DONE]\n\n");
        }

        private void frame(String text) {
            try {
                out.write(text.getBytes(StandardCharsets.UTF_8));
                out.flush();
            } catch (IOException e) {
                throw new UncheckedIOException(e); // client gone; unwound by guarded()
            }
        }

        @Override public void close() throws IOException {
            ACTIVE.remove(this);
            out.close();
        }
    }
}
