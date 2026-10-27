package com.qxotic.fetch;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.NoSuchFileException;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The contract's edges, on both transports: what is refused before a request, what a response looks
 * like when the server misbehaves, and what {@code follow} and {@code deadline} promise.
 */
class TransportQaTest {
    static HttpServer server;
    static URI base;
    static final Duration T = Duration.ofSeconds(10);
    static final AtomicInteger hits = new AtomicInteger();
    static final AtomicReference<String> authSeen = new AtomicReference<>();

    @BeforeAll
    static void serve() throws IOException {
        server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.createContext(
                "/multi",
                exchange -> {
                    hits.incrementAndGet();
                    exchange.getResponseHeaders().add("X-Multi", "one");
                    exchange.getResponseHeaders().add("X-Multi", "two");
                    String quote = exchange.getRequestHeaders().getFirst("X-Quote");
                    if (quote != null) exchange.getResponseHeaders().add("X-Quote-Echo", quote);
                    exchange.sendResponseHeaders(200, 2);
                    exchange.getResponseBody().write("ok".getBytes(StandardCharsets.US_ASCII));
                    exchange.close();
                });
        server.createContext(
                "/error",
                exchange -> {
                    exchange.sendResponseHeaders(500, 4);
                    exchange.getResponseBody().write("boom".getBytes(StandardCharsets.US_ASCII));
                    exchange.close();
                });
        server.createContext(
                "/truncated",
                exchange -> { // promises 100 bytes, delivers 50, hangs up
                    exchange.sendResponseHeaders(200, 100);
                    exchange.getResponseBody().write(new byte[50]);
                    exchange.getResponseBody().flush();
                    exchange.getHttpContext().getServer();
                    throw new IOException("hang up"); // the server closes the connection
                });
        server.createContext(
                "/loop",
                exchange -> {
                    exchange.getResponseHeaders().add("Location", "/loop");
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        server.createContext(
                "/nowhere",
                exchange -> {
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        server.createContext(
                "/badlocation",
                exchange -> {
                    exchange.getResponseHeaders().add("Location", "http://[bad");
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        server.createContext(
                "/auth",
                exchange -> {
                    authSeen.set(exchange.getRequestHeaders().getFirst("Authorization"));
                    exchange.sendResponseHeaders(204, -1);
                    exchange.close();
                });
        server.setExecutor(Executors.newCachedThreadPool());
        server.start();
        base = URI.create("http://127.0.0.1:" + server.getAddress().getPort());
    }

    @AfterAll
    static void stop() {
        server.stop(0);
    }

    static List<Transport> transports() {
        return CurlTransport.available()
                ? List.of(new JdkTransport(), new CurlTransport())
                : List.of(new JdkTransport());
    }

    // ---- refused before any request ----

    @Test
    void aUriThatIsNotHttpIsAFetchFailure() {
        for (Transport t : transports()) {
            for (String bad :
                    new String[] {"ftp://x/y", "/relative", "http:///nohost", "-o/etc/passwd"}) {
                IOException e =
                        assertThrows(
                                IOException.class, () -> t.get(URI.create(bad), Map.of(), T), bad);
                assertTrue(e.getMessage().contains("http(s)"), e.getMessage());
            }
        }
    }

    @Test
    void aHeaderWithALineBreakIsRefused() {
        int before = hits.get();
        for (Transport t : transports()) {
            for (Map<String, String> bad :
                    List.of(
                            Map.of("X-A", "one\r\nX-B: two"),
                            Map.of("X-A\nX-B", "x"),
                            Map.of("X-A: y", "x"))) {
                assertThrows(IOException.class, () -> t.get(base.resolve("/multi"), bad, T));
            }
        }
        assertEquals(before, hits.get(), "nothing was sent");
    }

    // ---- what a response says ----

    @Test
    void headersComeInOrderAndByAnyCase() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r = t.get(base.resolve("/multi"), Map.of(), T)) {
                assertEquals(List.of("one", "two"), r.headers("x-multi"));
                assertEquals("one", r.header("X-MULTI"));
                assertEquals(List.of(), r.headers("absent"));
                assertNull(r.header("absent"));
            }
        }
    }

    @Test
    void aHeaderValueWithQuotesAndBackslashesArrivesIntact() throws IOException {
        String value = "he said \"hi\" and wrote c:\\path\\to";
        for (Transport t : transports()) {
            try (Transport.Response r =
                    t.get(base.resolve("/multi"), Map.of("X-Quote", value), T)) {
                assertEquals(value, r.header("x-quote-echo"), t.getClass().getSimpleName());
            }
        }
    }

    @Test
    void aServerErrorIsAStatusWithItsBody() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r = t.get(base.resolve("/error"), Map.of(), T)) {
                assertEquals(500, r.status());
                assertEquals(
                        "boom", new String(r.body().readAllBytes(), StandardCharsets.US_ASCII));
            }
        }
    }

    @Test
    void aTruncatedBodyFailsBeforeItPassesForComplete() {
        for (Transport t : transports()) {
            assertThrows(
                    IOException.class,
                    () -> {
                        try (Transport.Response r = t.get(base.resolve("/truncated"), Map.of(), T);
                                InputStream in = r.body()) {
                            in.readAllBytes();
                        }
                    },
                    t.getClass().getSimpleName());
        }
    }

    @Test
    void theDefaultHeaderIsTheFirstValueOrNull() {
        Transport.Response r =
                new Transport.Response() {
                    @Override
                    public int status() {
                        return 200;
                    }

                    @Override
                    public List<String> headers(String name) {
                        return name.equals("x") ? List.of("a", "b") : List.of();
                    }

                    @Override
                    public InputStream body() {
                        return InputStream.nullInputStream();
                    }

                    @Override
                    public void close() {}
                };
        assertEquals("a", r.header("x"));
        assertNull(r.header("y"));
    }

    // ---- follow ----

    @Test
    void followStopsAtALoop() {
        for (Transport t : transports()) {
            IOException e =
                    assertThrows(
                            IOException.class, () -> t.follow(base.resolve("/loop"), Map.of(), T));
            assertTrue(e.getMessage().contains("redirect loop"), e.getMessage());
        }
    }

    @Test
    void followNeedsALocation() {
        for (Transport t : transports()) {
            IOException e =
                    assertThrows(
                            IOException.class,
                            () -> t.follow(base.resolve("/nowhere"), Map.of(), T));
            assertTrue(e.getMessage().contains("Location"), e.getMessage());
        }
    }

    @Test
    void followRefusesABadLocation() {
        for (Transport t : transports()) {
            IOException e =
                    assertThrows(
                            IOException.class,
                            () -> t.follow(base.resolve("/badlocation"), Map.of(), T));
            assertTrue(e.getMessage().contains("bad Location"), e.getMessage());
        }
    }

    @Test
    void followDropsTheCredentialWhenTheHostChanges() throws Exception {
        assumeTrue(
                InetAddress.getByName("localhost").getHostAddress().equals("127.0.0.1"),
                "localhost must be 127.0.0.1 for a second host name on one server");
        URI other = URI.create("http://localhost:" + server.getAddress().getPort() + "/auth");
        server.createContext(
                "/away",
                exchange -> {
                    exchange.getResponseHeaders().add("Location", other.toString());
                    exchange.sendResponseHeaders(302, -1);
                    exchange.close();
                });
        for (Transport t : transports()) {
            authSeen.set("unset");
            try (Transport.Response r =
                    t.follow(base.resolve("/away"), Map.of("Authorization", "Bearer z"), T)) {
                assertEquals(204, r.status());
            }
            assertNull(
                    authSeen.get(),
                    t.getClass().getSimpleName() + ": another host never sees the token");
        }
    }

    // ---- 1xx, which the JDK's server cannot send: a raw socket, for the parser curl's output
    // feeds ----

    @Test
    void curlSkipsAnInformationalResponse() throws Exception {
        assumeTrue(CurlTransport.available());
        try (ServerSocket raw = new ServerSocket(0, 1, InetAddress.getLoopbackAddress())) {
            Thread responder =
                    new Thread(
                            () -> {
                                try (Socket s = raw.accept();
                                        OutputStream out = s.getOutputStream()) {
                                    s.getInputStream().read(new byte[4096]);
                                    out.write(
                                            ("HTTP/1.1 103 Early Hints\r\n"
                                                            + "Link: </x>; rel=preload\r\n\r\n"
                                                            + "HTTP/1.1 200 OK\r\n"
                                                            + "Content-Length: 2\r\n"
                                                            + "X-Real: yes\r\n\r\n"
                                                            + "ok")
                                                    .getBytes(StandardCharsets.US_ASCII));
                                } catch (IOException ignored) {
                                    // the client closed first
                                }
                            });
            responder.setDaemon(true);
            responder.start();
            URI url = URI.create("http://127.0.0.1:" + raw.getLocalPort() + "/hints");
            try (Transport.Response r = new CurlTransport().get(url, Map.of(), T)) {
                assertEquals(200, r.status());
                assertEquals("yes", r.header("x-real"));
                assertNull(r.header("link"), "the 1xx block's headers are not the response's");
                assertEquals("ok", new String(r.body().readAllBytes(), StandardCharsets.US_ASCII));
            }
        }
    }

    // ---- deadline and system ----

    @Test
    void aDeadlineClosedInTimeLeavesNothingScheduled() throws IOException {
        for (Transport t : transports()) {
            try (Transport.Response r =
                    Transport.deadline(t.get(base.resolve("/multi"), Map.of(), T), T)) {
                assertEquals("ok", new String(r.body().readAllBytes(), StandardCharsets.US_ASCII));
            }
        }
        assertTrue(
                Transports.DEADLINES.getQueue().isEmpty(), "a cancelled deadline leaves the queue");
    }

    @Test
    void offlineRefusesEveryRequestBeforeConnecting() {
        int before = hits.get();
        NoSuchFileException e =
                assertThrows(
                        NoSuchFileException.class,
                        () -> Transport.offline().get(base.resolve("/multi"), Map.of(), T));
        assertTrue(e.getMessage().startsWith("offline"), e.getMessage());
        assertThrows(
                NoSuchFileException.class,
                () -> Transport.offline().follow(base.resolve("/multi"), Map.of(), T));
        assertEquals(before, hits.get(), "nothing was sent");
    }

    @Test
    void theSystemTransportIsChosenOnce() {
        assertSame(Transport.system(), Transport.system());
    }
}
