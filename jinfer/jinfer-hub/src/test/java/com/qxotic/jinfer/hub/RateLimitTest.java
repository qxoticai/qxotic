package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.net.http.HttpHeaders;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * A 429 is the server saying WHEN, not a failure. Seen on the Hugging Face CDN during a large
 * anonymous pull: {@code RateLimit: "resolvers";r=0;t=21} under a 3000-requests-per-300-s policy,
 * after which three instant retries failed the whole pull within a second.
 */
class RateLimitTest {

    private static HttpHeaders headers(Map<String, List<String>> map) {
        return HttpHeaders.of(map, (name, value) -> true);
    }

    @Test
    void readsTheResetOfTheExhaustedHuggingFacePolicy() {
        HttpHeaders h =
                headers(
                        Map.of(
                                "RateLimit", List.of("\"resolvers\";r=0;t=21"),
                                "RateLimit-Policy",
                                        List.of("\"fixed window\";\"resolvers\";q=3000;w=300")));
        assertEquals(Duration.ofSeconds(21), RateLimit.requested(h, 0));
    }

    @Test
    void ignoresPoliciesThatStillHaveRoom() {
        HttpHeaders h =
                headers(Map.of("RateLimit", List.of("\"api\";r=5;t=90, \"resolvers\";r=0;t=7")));
        assertEquals(Duration.ofSeconds(7), RateLimit.requested(h, 0));
        assertNull(
                RateLimit.requested(headers(Map.of("RateLimit", List.of("\"api\";r=5;t=90"))), 0));
    }

    @Test
    void retryAfterWinsInSecondsOrAsADate() {
        assertEquals(
                Duration.ofSeconds(30),
                RateLimit.requested(
                        headers(
                                Map.of(
                                        "Retry-After",
                                        List.of("30"),
                                        "RateLimit",
                                        List.of("\"x\";r=0;t=5"))),
                        0));
        long now = java.time.ZonedDateTime.parse("2026-09-14T12:00:00Z").toInstant().toEpochMilli();
        assertEquals(
                Duration.ofSeconds(90),
                RateLimit.requested(
                        headers(Map.of("Retry-After", List.of("Mon, 14 Sep 2026 12:01:30 GMT"))),
                        now));
    }

    @Test
    void understandsTheOlderDraftForms() {
        assertEquals(
                Duration.ofSeconds(12),
                RateLimit.requested(
                        headers(Map.of("RateLimit", List.of("limit=100, remaining=0, reset=12"))),
                        0));
        assertEquals(
                Duration.ofSeconds(4),
                RateLimit.requested(headers(Map.of("RateLimit-Reset", List.of("4"))), 0));
        assertNull(RateLimit.requested(headers(Map.of()), 0));
    }

    @Test
    void aRateLimitedDownloadWaitsWhatTheServerAskedAndCompletes(@TempDir Path dir)
            throws IOException {
        byte[] payload = new byte[150_000];
        new Random(3).nextBytes(payload);
        AtomicInteger hits = new AtomicInteger();
        HttpServer server = HttpServer.create(new InetSocketAddress("127.0.0.1", 0), 0);
        server.setExecutor(
                Executors.newCachedThreadPool(Thread.ofPlatform().daemon(true).factory()));
        server.createContext(
                "/rl.bin",
                exchange -> {
                    if (hits.incrementAndGet() <= 2) { // the window is spent: say when it resets
                        exchange.getResponseHeaders().add("RateLimit", "\"resolvers\";r=0;t=1");
                        exchange.sendResponseHeaders(429, -1);
                        exchange.close();
                        return;
                    }
                    exchange.sendResponseHeaders(200, payload.length);
                    try (OutputStream out = exchange.getResponseBody()) {
                        out.write(payload);
                    }
                });
        server.start();
        try {
            String url = "http://127.0.0.1:" + server.getAddress().getPort() + "/rl.bin";
            Path dest = dir.resolve("rl.bin");
            long t0 = System.nanoTime();
            Fetch.download(url, dest, payload.length, null, Map.of());
            double seconds = (System.nanoTime() - t0) / 1e9;
            assertArrayEquals(payload, Files.readAllBytes(dest));
            assertTrue(seconds >= 1.0, "did not wait for the window: " + seconds + " s");
            assertTrue(seconds < 20, "waited far longer than asked: " + seconds + " s");
            assertEquals(3, hits.get(), "a held host must not be asked again before it clears");
        } finally {
            server.stop(0);
        }
    }
}
