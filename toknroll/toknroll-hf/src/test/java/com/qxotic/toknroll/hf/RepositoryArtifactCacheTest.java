package com.qxotic.toknroll.hf;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTimeoutPreemptively;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RepositoryArtifactCacheTest {
    @TempDir Path tempDir;
    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void fetchUrl_usesCacheWhenPresent() throws Exception {
        AtomicInteger hits = new AtomicInteger();
        startServer("/artifact", exchange -> writeResponse(exchange, hits.incrementAndGet(), 200));

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/artifact";

        Path first = cache.fetchUrl(url, Map.of(), false, false);
        Path second = cache.fetchUrl(url, Map.of(), false, false);

        assertEquals(first, second);
        assertEquals(1, hits.get(), "second fetch should reuse cache");
        assertEquals("1", Files.readString(first, StandardCharsets.UTF_8));
    }

    @Test
    void fetchUrl_offlineMissThrows() throws Exception {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        IOException error =
                assertThrows(
                        IOException.class,
                        () ->
                                cache.fetchUrl(
                                        "https://example.com/never-used", Map.of(), true, false));
        assertTrue(error.getMessage().contains("useCacheOnly=true"));
    }

    @Test
    void fetchUrl_forceRefreshRedownloads() throws Exception {
        AtomicInteger hits = new AtomicInteger();
        startServer("/refresh", exchange -> writeResponse(exchange, hits.incrementAndGet(), 200));

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/refresh";

        Path first = cache.fetchUrl(url, Map.of(), false, false);
        Path second = cache.fetchUrl(url, Map.of(), false, true);

        assertEquals(first, second);
        assertEquals(2, hits.get(), "forceRefresh should trigger second download");
        assertEquals("2", Files.readString(second, StandardCharsets.UTF_8));
    }

    @Test
    void fetchUrl_cachesBinaryPayloadUnchanged() throws Exception {
        byte[] payload =
                new byte[] {0x00, (byte) 0xFF, 0x10, 0x00, 0x7F, (byte) 0x80, 0x55, (byte) 0xAA};
        startServer(
                "/binary",
                exchange -> {
                    exchange.sendResponseHeaders(200, payload.length);
                    try (OutputStream out = exchange.getResponseBody()) {
                        out.write(payload);
                    }
                });

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/binary";

        Path path = cache.fetchUrl(url, Map.of(), false, false);
        assertArrayEquals(payload, Files.readAllBytes(path));
    }

    @Test
    void fetchUrl_appliesHeaders() throws Exception {
        startServer(
                "/auth",
                exchange -> {
                    List<String> values = exchange.getRequestHeaders().get("Authorization");
                    if (values == null
                            || values.isEmpty()
                            || !"Bearer secret".equals(values.get(0))) {
                        writeResponse(exchange, "forbidden", 403);
                        return;
                    }
                    writeResponse(exchange, "ok", 200);
                });

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/auth";

        Path path =
                cache.fetchUrl(
                        url, Map.of("Authorization", List.of("Bearer secret")), false, false);
        assertEquals("ok", Files.readString(path, StandardCharsets.UTF_8));
    }

    @Test
    void fetchUrl_cleansUpPartialOnFailure() throws Exception {
        startServer("/error", exchange -> writeResponse(exchange, "error", 500));

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/error";

        assertThrows(IOException.class, () -> cache.fetchUrl(url, Map.of(), false, false));
        assertNoPartials();
    }

    @Test
    void fetchUrl_concurrentDownloadsDoNotSharePartialFile() throws Exception {
        CountDownLatch requests = new CountDownLatch(2);
        startServer(
                "/concurrent",
                exchange -> {
                    requests.countDown();
                    await(requests);
                    writeResponse(exchange, "complete", 200);
                });

        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        String url = "http://localhost:" + server.getAddress().getPort() + "/concurrent";
        ExecutorService callers = Executors.newFixedThreadPool(2);
        try {
            Future<Path> first = callers.submit(() -> cache.fetchUrl(url, Map.of(), false, true));
            Future<Path> second = callers.submit(() -> cache.fetchUrl(url, Map.of(), false, true));

            assertEquals(first.get(2, TimeUnit.SECONDS), second.get(2, TimeUnit.SECONDS));
            assertEquals("complete", Files.readString(first.get(), StandardCharsets.UTF_8));
        } finally {
            callers.shutdownNow();
        }
        assertNoPartials();
    }

    @Test
    void fetchUrl_timesOutStalledBody() throws Exception {
        startServer(
                "/stalled",
                exchange -> {
                    exchange.sendResponseHeaders(200, 1);
                    await(new CountDownLatch(1));
                });

        RepositoryArtifactCache cache =
                RepositoryArtifactCache.create(tempDir, Duration.ofMillis(100));
        String url = "http://localhost:" + server.getAddress().getPort() + "/stalled";

        assertTimeoutPreemptively(
                Duration.ofSeconds(2),
                () ->
                        assertThrows(
                                IOException.class,
                                () -> cache.fetchUrl(url, Map.of(), false, false)));
        assertNoPartials();
    }

    @Test
    void fetchHuggingFace_blankRevisionRejected() {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchHuggingFace("user", "repo", "   ", "tokenizer.json", true, false));
    }

    @Test
    void fetchModelScope_blankRevisionRejected() {
        RepositoryArtifactCache cache = RepositoryArtifactCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchModelScope("user", "repo", "   ", "tokenizer.json", true, false));
    }

    private void startServer(String route, ThrowingHandler handler) throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.setExecutor(
                Executors.newCachedThreadPool(
                        task -> {
                            Thread thread = new Thread(task);
                            thread.setDaemon(true);
                            return thread;
                        }));
        server.createContext(
                route,
                exchange -> {
                    try {
                        handler.handle(exchange);
                    } finally {
                        exchange.close();
                    }
                });
        server.start();
    }

    private void assertNoPartials() throws IOException {
        try (var files = Files.walk(tempDir)) {
            assertTrue(files.noneMatch(path -> path.getFileName().toString().endsWith(".partial")));
        }
    }

    private static void await(CountDownLatch latch) throws IOException {
        try {
            if (!latch.await(2, TimeUnit.SECONDS)) {
                throw new IOException("Timed out waiting for test peer");
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IOException(e);
        }
    }

    private static void writeResponse(HttpExchange exchange, Object body, int status)
            throws IOException {
        byte[] bytes = String.valueOf(body).getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    @FunctionalInterface
    private interface ThrowingHandler {
        void handle(HttpExchange exchange) throws IOException;
    }
}
