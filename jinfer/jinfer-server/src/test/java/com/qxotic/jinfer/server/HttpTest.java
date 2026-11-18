package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.logging.Handler;
import java.util.logging.LogRecord;
import java.util.logging.Logger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

class HttpTest {

    /** One line per exchange, written once it is answered: method, path, status, duration. */
    @Test
    void theAccessLogLineCarriesStatusAndDuration() throws Exception {
        var logger = Logger.getLogger("jinfer.server");
        var lines = new CopyOnWriteArrayList<String>();
        var capture =
                new Handler() {
                    @Override
                    public void publish(LogRecord record) {
                        lines.add(record.getMessage());
                    }

                    @Override
                    public void flush() {}

                    @Override
                    public void close() {}
                };
        logger.addHandler(capture);
        var server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        try {
            Http.logged(
                    server.createContext(
                            "/teapot",
                            exchange -> Http.sendError(exchange, 418, "short and stout")));
            server.start();
            var client = HttpClient.newHttpClient();
            var response =
                    client.send(
                            HttpRequest.newBuilder(
                                            URI.create(
                                                    "http://127.0.0.1:"
                                                            + server.getAddress().getPort()
                                                            + "/teapot"))
                                    .build(),
                            HttpResponse.BodyHandlers.discarding());
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
    void headIsGetWithoutTheBody() throws Exception {
        // RFC 9110: wherever GET works, HEAD does; uptime monitors probe with it
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/",
                exchange -> {
                    if (Http.requireMethod(exchange, "GET")) return;
                    Http.sendJson(exchange, 200, java.util.Map.of("status", "ok"));
                });
        server.start();
        try (HttpClient client = HttpClient.newHttpClient()) {
            URI uri = URI.create("http://127.0.0.1:" + server.getAddress().getPort() + "/");
            HttpResponse<String> head =
                    client.send(
                            HttpRequest.newBuilder(uri).HEAD().build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(200, head.statusCode());
            assertEquals("", head.body());
            assertTrue(
                    head.headers()
                            .firstValue("Content-Type")
                            .orElse("")
                            .startsWith("application/json"));
            HttpResponse<String> post =
                    client.send(
                            HttpRequest.newBuilder(uri)
                                    .POST(HttpRequest.BodyPublishers.noBody())
                                    .build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(405, post.statusCode());
            assertEquals("GET, HEAD, OPTIONS", post.headers().firstValue("Allow").orElse(""));
        } finally {
            server.stop(0);
        }
    }

    @Test
    @Timeout(30)
    void anUploadFarPastTheLimitStillReadsItsFourThirteen() throws Exception {
        // a client that sends its whole request before reading, as most do: closing with the
        // upload still arriving reset the connection, and it never saw the 413 that says why
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/", exchange -> Http.readBody(exchange, 1024, Duration.ofSeconds(20)));
        server.start();
        try (Socket client = new Socket("127.0.0.1", server.getAddress().getPort())) {
            client.setSoTimeout(20_000);
            byte[] body = new byte[8 << 20];
            OutputStream out = client.getOutputStream();
            out.write(
                    ("POST / HTTP/1.1\r\nHost: localhost\r\nContent-Length: "
                                    + body.length
                                    + "\r\n\r\n")
                            .getBytes(StandardCharsets.US_ASCII));
            out.write(body);
            out.flush();
            InputStream in = client.getInputStream();
            StringBuilder response = new StringBuilder();
            byte[] chunk = new byte[4096];
            for (int n;
                    !response.toString().contains("1024-byte limit") && (n = in.read(chunk)) > 0; )
                response.append(new String(chunk, 0, n, StandardCharsets.UTF_8));
            assertTrue(response.toString().startsWith("HTTP/1.1 413"), response.toString());
            assertTrue(response.toString().contains("1024-byte limit"), response.toString());
        } finally {
            server.stop(0);
        }
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
