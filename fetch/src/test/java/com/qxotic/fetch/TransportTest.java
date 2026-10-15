package com.qxotic.fetch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.net.InetSocketAddress;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/** Both transports against one local server: the same answers, byte for byte. */
class TransportTest {
    static HttpServer server;
    static URI base;
    static final AtomicReference<String> lastAuthorization = new AtomicReference<>();
    static final Duration T = Duration.ofSeconds(10);
    static final byte[] BODY = "0123456789abcdef".repeat(1000).getBytes(StandardCharsets.US_ASCII);

    @BeforeAll
    static void serve() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/file",
                exchange -> {
                    lastAuthorization.set(exchange.getRequestHeaders().getFirst("Authorization"));
                    String range = exchange.getRequestHeaders().getFirst("Range");
                    exchange.getResponseHeaders().add("X-Echo", "yes");
                    if (range != null) {
                        exchange.getResponseHeaders()
                                .add("Content-Range", "bytes 0-9/" + BODY.length);
                        exchange.sendResponseHeaders(206, 10);
                        exchange.getResponseBody().write(BODY, 0, 10);
                    } else {
                        exchange.sendResponseHeaders(200, BODY.length);
                        exchange.getResponseBody().write(BODY);
                    }
                    exchange.close();
                });
        server.createContext(
                "/redirect",
                exchange -> {
                    exchange.getResponseHeaders().add("Location", "/file");
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        server.createContext(
                "/missing",
                exchange -> {
                    exchange.sendResponseHeaders(404, -1);
                    exchange.close();
                });
        server.setExecutor(
                java.util.concurrent.Executors
                        .newCachedThreadPool()); // a sleeping handler must not block the next
        // request
        server.start();
        base = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    @AfterAll
    static void stop() {
        server.stop(0);
    }

    static List<Transport> transports() {
        return CurlTransport.available()
                ? List.of(new JdkTransport(), Transport.curl())
                : List.of(new JdkTransport());
    }

    @Test
    void theWholeBodyAndItsHeaders() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r =
                    t.get(base.resolve("/file"), Map.of("Authorization", "Bearer x"), T)) {
                assertEquals(200, r.status(), t.getClass().getSimpleName());
                assertEquals("yes", r.header("x-echo"));
                assertEquals("yes", r.header("X-Echo"));
                assertNull(r.header("nope"));
                assertEquals(BODY.length, r.body().readAllBytes().length);
            }
            assertEquals("Bearer x", lastAuthorization.get());
        }
    }

    @Test
    void aRangeIsAHeaderInAndOut() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r =
                    t.get(base.resolve("/file"), Map.of("Range", "bytes=0-9"), T)) {
                assertEquals(206, r.status(), t.getClass().getSimpleName());
                assertEquals("bytes 0-9/" + BODY.length, r.header("content-range"));
                assertEquals(
                        "0123456789",
                        new String(r.body().readAllBytes(), StandardCharsets.US_ASCII));
            }
        }
    }

    @Test
    void getReportsARedirectAndFollowTakesIt() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r = t.get(base.resolve("/redirect"), Map.of(), T)) {
                assertEquals(302, r.status(), t.getClass().getSimpleName());
                assertEquals("/file", r.header("location"));
            }
            try (Transport.Response r =
                    t.follow(base.resolve("/redirect"), Map.of("Authorization", "Bearer y"), T)) {
                assertEquals(200, r.status());
                assertEquals(BODY.length, r.body().readAllBytes().length);
            }
            assertEquals("Bearer y", lastAuthorization.get(), "same host: the credential follows");
        }
    }

    @Test
    void aStatusIsAStatusNotAnException() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r = t.get(base.resolve("/missing"), Map.of(), T)) {
                assertEquals(404, r.status(), t.getClass().getSimpleName());
            }
        }
    }

    @Test
    void curlNamesAConnectionItCouldNotMake() {
        assumeTrue(CurlTransport.available());
        IOException e =
                assertThrows(
                        IOException.class,
                        () ->
                                Transport.curl()
                                        .get(URI.create("http://127.0.0.1:9/x"), Map.of(), T));
        assertEquals(
                true,
                e.getMessage().toLowerCase().contains("curl")
                        || e.getMessage().toLowerCase().contains("connect"),
                e.getMessage());
    }

    @Test
    void theSystemTransportIsTheJdkClientWhenItIsOnTheClassPath() {
        assertEquals(JdkTransport.class, Transport.system().getClass());
    }

    @Test
    void aServerThatNeverAnswersIsATimeoutNotAHang() throws IOException {
        server.createContext(
                "/stalled",
                exchange -> {
                    try {
                        Thread.sleep(30_000);
                    } catch (InterruptedException ignored) {
                    }
                    exchange.close();
                });
        for (Transport t : transports()) {
            long start = System.nanoTime();
            assertThrows(
                    IOException.class,
                    () -> t.get(base.resolve("/stalled"), Map.of(), Duration.ofMillis(300)).close(),
                    t.getClass().getSimpleName());
            assertEquals(true, System.nanoTime() - start < 5_000_000_000L, "gave up within 5 s");
        }
    }

    @Test
    void aBodyThatStallsIsClosedByItsDeadline() throws IOException {
        // a raw socket: the JDK's HttpServer holds the headers back until it has body to send with
        // them
        try (java.net.ServerSocket drip =
                new java.net.ServerSocket(0, 1, java.net.InetAddress.getLoopbackAddress())) {
            Thread stall =
                    new Thread(
                            () -> {
                                try {
                                    while (true) {
                                        java.net.Socket c = drip.accept();
                                        Thread one =
                                                new Thread(
                                                        () -> { // one connection per thread: the
                                                            // stall must not starve the next
                                                            // transport
                                                            try {
                                                                c.getInputStream()
                                                                        .read(new byte[4096]);
                                                                c.getOutputStream()
                                                                        .write(
                                                                                "HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\nx"
                                                                                        .getBytes(
                                                                                                StandardCharsets
                                                                                                        .US_ASCII));
                                                                c.getOutputStream().flush();
                                                                Thread.sleep(30_000);
                                                                c.close();
                                                            } catch (Exception ignored) {
                                                                // closed under it
                                                            }
                                                        });
                                        one.setDaemon(true);
                                        one.start();
                                    }
                                } catch (Exception ignored) {
                                    // the server socket closed with the test
                                }
                            });
            stall.setDaemon(true);
            stall.start();
            URI url = URI.create("http://127.0.0.1:" + drip.getLocalPort() + "/drip");
            for (Transport t : transports()) {
                long start = System.nanoTime();
                assertThrows(
                        IOException.class,
                        () -> {
                            try (Transport.Response r =
                                    Transport.deadline(
                                            t.get(url, Map.of(), T), Duration.ofMillis(300))) {
                                assertEquals(200, r.status());
                                r.body().readAllBytes();
                            }
                        },
                        t.getClass().getSimpleName());
                long ms = (System.nanoTime() - start) / 1_000_000;
                assertEquals(
                        true,
                        ms < 5_000,
                        t.getClass().getSimpleName() + " gave up after " + ms + " ms");
            }
        }
    }

    @Test
    void curlRefusesAUrlThatCouldReadAsAnOption() {
        assumeTrue(CurlTransport.available());
        assertThrows(
                IOException.class,
                () -> Transport.curl().get(URI.create("-o/etc/passwd"), Map.of(), T));
    }

    @Test
    void anEmptyBodyReadsAsEmpty() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r = t.get(base.resolve("/missing"), Map.of(), T);
                    InputStream in = r.body()) {
                assertEquals(-1, in.read(), t.getClass().getSimpleName());
            }
        }
    }
}
