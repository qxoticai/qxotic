package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CountDownLatch;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class HttpTest {

    /** One line per exchange, written once it is answered: method, path, status, duration. */
    @Test
    void theAccessLogLineCarriesStatusAndDuration() throws Exception {
        var logger = java.util.logging.Logger.getLogger("jinfer.server");
        var lines = new java.util.concurrent.CopyOnWriteArrayList<String>();
        var capture =
                new java.util.logging.Handler() {
                    @Override
                    public void publish(java.util.logging.LogRecord record) {
                        lines.add(record.getMessage());
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        logger.addHandler(capture);
        var server =
                com.sun.net.httpserver.HttpServer.create(
                        new java.net.InetSocketAddress("127.0.0.1", 0), 0);
        try {
            Http.logged(
                    server.createContext(
                            "/teapot",
                            exchange -> Http.sendError(exchange, 418, "short and stout")));
            server.start();
            var client = java.net.http.HttpClient.newHttpClient();
            var response =
                    client.send(
                            java.net.http.HttpRequest.newBuilder(
                                            java.net.URI.create(
                                                    "http://127.0.0.1:"
                                                            + server.getAddress().getPort()
                                                            + "/teapot"))
                                    .build(),
                            java.net.http.HttpResponse.BodyHandlers.discarding());
            assertEquals(418, response.statusCode());
            // the line follows the response: the client can read it before the filter returns
            long deadline = System.nanoTime() + Duration.ofSeconds(5).toNanos();
            while (lines.isEmpty() && System.nanoTime() < deadline) Thread.onSpinWait();
            assertTrue(
                    lines.stream().anyMatch(l -> l.matches("GET /teapot 418 \\d+ ms from .+")),
                    lines.toString());
        } finally {
            server.stop(0);
            logger.removeHandler(capture);
        }
    }

    @Test
    void bodyLimitAcceptsTheBoundaryAndRejectsTheNextByte() throws Exception {
        TestExchange exact = new TestExchange("1234".getBytes(StandardCharsets.UTF_8));
        assertArrayEquals(
                "1234".getBytes(StandardCharsets.UTF_8),
                Http.readBody(exact, 4, Duration.ofSeconds(1)));
        assertEquals(-1, exact.getResponseCode());

        TestExchange oversized = new TestExchange("12345".getBytes(StandardCharsets.UTF_8));
        assertNull(Http.readBody(oversized, 4, Duration.ofSeconds(1)));
        assertEquals(413, oversized.getResponseCode());
        assertTrue(
                new String(oversized.responseBytes(), StandardCharsets.UTF_8).contains("4-byte"));
    }

    @Test
    @Timeout(2)
    void bodyReadDeadlineClosesAStalledExchange() throws Exception {
        CountDownLatch closed = new CountDownLatch(1);
        InputStream stalled =
                new InputStream() {
                    @Override
                    public int read() throws IOException {
                        try {
                            closed.await();
                            return -1;
                        } catch (InterruptedException e) {
                            Thread.currentThread().interrupt();
                            throw new IOException(e);
                        }
                    }

                    @Override
                    public void close() {
                        closed.countDown();
                    }
                };
        TestExchange exchange = new TestExchange(stalled, new ByteArrayOutputStream());

        assertNull(Http.readBody(exchange, 4, Duration.ofMillis(20)));
        assertTrue(exchange.closed());
    }

    @Test
    @Timeout(2)
    void completedBodyReadCancelsItsDeadline() throws Exception {
        TestExchange exchange = new TestExchange("{}".getBytes(StandardCharsets.UTF_8));

        assertArrayEquals(
                "{}".getBytes(StandardCharsets.UTF_8),
                Http.readBody(exchange, 4, Duration.ofMillis(100)));
        Thread.sleep(200);

        assertFalse(exchange.closed());
    }

    @Test
    void errorEnvelopeTypesFollowTheOpenAiSpelling() {
        assertEquals("authentication_error", Http.errorPayload(401, "no", null).get("type"));
        assertEquals("rate_limit_error", Http.errorPayload(429, "slow", null).get("type"));
        assertEquals("server_error", Http.errorPayload(503, "busy", null).get("type"));
        assertEquals("invalid_request_error", Http.errorPayload(400, "bad", null).get("type"));
    }
}
