package com.qxotic.jinfer.x.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;

class SseTest {

    @Test
    void framesAreFlushedNumberedAndTerminated() throws Exception {
        TestExchange exchange = new TestExchange(new byte[0]);
        try (Sse.Stream stream = Sse.begin(exchange, Duration.ofSeconds(1))) {
            stream.emit("first", Map.of("type", "first"));
            stream.emit("second", Map.of("type", "second"));
            stream.done();
        }

        String body = new String(exchange.responseBytes(), StandardCharsets.UTF_8);
        assertEquals(200, exchange.getResponseCode());
        assertEquals(
                "text/event-stream; charset=utf-8",
                exchange.getResponseHeaders().getFirst("Content-Type"));
        assertTrue(
                body.contains(
                        "event: first\ndata: {\"type\":\"first\",\"sequence_number\":0}"),
                body);
        assertTrue(
                body.contains(
                        "event: second\ndata: {\"type\":\"second\",\"sequence_number\":1}"),
                body);
        assertTrue(body.endsWith("data: [DONE]\n\n"), body);
    }

    @Test
    void aDisconnectedClientUnwindsAsIOException() throws Exception {
        OutputStream disconnected =
                new OutputStream() {
                    @Override
                    public void write(int value) throws IOException {
                        throw new IOException("client gone");
                    }
                };
        TestExchange exchange =
                new TestExchange(new ByteArrayInputStream(new byte[0]), disconnected);
        try (Sse.Stream stream = Sse.begin(exchange, Duration.ofSeconds(1))) {
            IOException failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Sse.guarded(
                                            stream,
                                            new Metrics(),
                                            () -> stream.emit(Map.of("x", 1))));
            assertEquals("client gone", failure.getMessage());
        }
    }

    @Test
    void theReaperClosesAStalledWriter() throws Exception {
        BlockingOutput output = new BlockingOutput();
        TestExchange exchange =
                new TestExchange(new ByteArrayInputStream(new byte[0]), output);
        AtomicReference<Throwable> failure = new AtomicReference<>();
        Sse.startReaper();

        Thread writer =
                Thread.ofPlatform()
                        .start(
                                () -> {
                                    try (Sse.Stream stream =
                                            Sse.begin(exchange, Duration.ofMillis(10))) {
                                        Sse.guarded(
                                                stream,
                                                new Metrics(),
                                                () -> stream.emit(Map.of("x", 1)));
                                    } catch (Throwable t) {
                                        failure.set(t);
                                    }
                                });

        assertTrue(output.started.await(2, TimeUnit.SECONDS), "write never started");
        assertTrue(output.closed.await(3, TimeUnit.SECONDS), "reaper did not close the stream");
        writer.join(2_000);
        assertTrue(!writer.isAlive(), "writer remained blocked");
        assertTrue(exchange.closed());
        assertInstanceOf(IOException.class, failure.get());
    }

    private static final class BlockingOutput extends OutputStream {
        private final CountDownLatch started = new CountDownLatch(1);
        private final CountDownLatch closed = new CountDownLatch(1);

        @Override
        public void write(int value) throws IOException {
            started.countDown();
            try {
                closed.await();
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                throw new IOException("interrupted", e);
            }
            throw new IOException("closed");
        }

        @Override
        public void close() {
            closed.countDown();
        }
    }
}
