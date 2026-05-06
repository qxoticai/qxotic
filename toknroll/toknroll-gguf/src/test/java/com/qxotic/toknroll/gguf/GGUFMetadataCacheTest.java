package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.Builder;
import com.qxotic.format.gguf.GGUF;
import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;
import java.io.IOException;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GGUFMetadataCacheTest {

    @TempDir Path tempDir;
    private HttpServer server;

    @AfterEach
    void tearDown() {
        if (server != null) {
            server.stop(0);
        }
    }

    @Test
    void fetchHuggingFace_usesCachedMetadataWhenPresent() throws Exception {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        Path cachedPath =
                tempDir.resolve("gguf-metadata")
                        .resolve("huggingface")
                        .resolve("unsloth")
                        .resolve("gemma-4-E2B-it-GGUF")
                        .resolve("main")
                        .resolve("gemma-4-E2B-it-Q8_0.gguf.metadata");
        Files.createDirectories(cachedPath.getParent());
        Files.writeString(cachedPath, "cached", StandardCharsets.UTF_8);

        Path actual =
                cache.fetchHuggingFace(
                        "unsloth",
                        "gemma-4-E2B-it-GGUF",
                        null,
                        "gemma-4-E2B-it-Q8_0.gguf",
                        true,
                        false);

        assertEquals(cachedPath.toAbsolutePath().normalize(), actual);
        assertTrue(actual.toString().contains("/main/"));
        assertTrue(actual.toString().endsWith(".gguf.metadata"));
    }

    @Test
    void fetchHuggingFace_cachedNestedPathUsesMetadataSuffix() throws Exception {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        Path cachedPath =
                tempDir.resolve("gguf-metadata")
                        .resolve("huggingface")
                        .resolve("unsloth")
                        .resolve("Qwen3.6-35B-A3B-GGUF")
                        .resolve("main")
                        .resolve("quantized")
                        .resolve("Qwen3.6-35B-A3B-Q8_0.gguf.metadata");
        Files.createDirectories(cachedPath.getParent());
        Files.writeString(cachedPath, "cached", StandardCharsets.UTF_8);

        Path actual =
                cache.fetchHuggingFace(
                        "unsloth",
                        "Qwen3.6-35B-A3B-GGUF",
                        null,
                        "quantized/Qwen3.6-35B-A3B-Q8_0.gguf",
                        true,
                        false);

        assertEquals(cachedPath.toAbsolutePath().normalize(), actual);
        assertTrue(actual.toString().endsWith(".gguf.metadata"));
    }

    @Test
    void fetchModelScope_usesCachedMetadataWhenPresent() throws Exception {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        Path cachedPath =
                tempDir.resolve("gguf-metadata")
                        .resolve("modelscope")
                        .resolve("Qwen")
                        .resolve("Qwen3.6-35B-A3B-GGUF")
                        .resolve("master")
                        .resolve("Qwen3.6-35B-A3B-Q8_0.gguf.metadata");
        Files.createDirectories(cachedPath.getParent());
        Files.writeString(cachedPath, "cached", StandardCharsets.UTF_8);

        Path actual =
                cache.fetchModelScope(
                        "Qwen",
                        "Qwen3.6-35B-A3B-GGUF",
                        null,
                        "Qwen3.6-35B-A3B-Q8_0.gguf",
                        true,
                        false);

        assertEquals(cachedPath.toAbsolutePath().normalize(), actual);
        assertTrue(actual.toString().contains("/master/"));
        assertTrue(actual.toString().endsWith(".gguf.metadata"));
    }

    @Test
    void fetchHuggingFace_offlineMissThrows() {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);

        IOException error =
                assertThrows(
                        IOException.class,
                        () ->
                                cache.fetchHuggingFace(
                                        "unsloth",
                                        "Llama-3.2-1B-Instruct-GGUF",
                                        null,
                                        "Llama-3.2-1B-Instruct-Q8_0.gguf",
                                        true,
                                        false));

        assertTrue(error.getMessage().contains("useCacheOnly=true"));
    }

    @Test
    void fetchHuggingFace_blankRevisionRejected() {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        cache.fetchHuggingFace(
                                "unsloth",
                                "gpt-oss-20b-GGUF",
                                "   ",
                                "gpt-oss-20b-Q8_0.gguf",
                                true,
                                false));
    }

    @Test
    void fetchModelScope_blankRevisionRejected() {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        cache.fetchModelScope(
                                "Qwen",
                                "Qwen3.6-35B-A3B-GGUF",
                                "   ",
                                "Qwen3.6-35B-A3B-Q8_0.gguf",
                                true,
                                false));
    }

    @Test
    void fetchHuggingFace_nonGgufPathRejected() {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchHuggingFace("a", "b", null, "tokenizer.json", true, false));
    }

    @Test
    void fetchHuggingFace_invalidSegmentsRejected() {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchHuggingFace("a/b", "repo", null, "model.gguf", true, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchHuggingFace("user", "../repo", null, "model.gguf", true, false));
        assertThrows(
                IllegalArgumentException.class,
                () -> cache.fetchHuggingFace("user", "repo", null, "../model.gguf", true, false));
    }

    @Test
    void fetchHuggingFace_forceRefreshWithCacheOnlyThrows() throws Exception {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        Path cachedPath =
                tempDir.resolve("gguf-metadata")
                        .resolve("huggingface")
                        .resolve("unsloth")
                        .resolve("Llama-3.2-1B-Instruct-GGUF")
                        .resolve("main")
                        .resolve("Llama-3.2-1B-Instruct-Q8_0.gguf.metadata");
        Files.createDirectories(cachedPath.getParent());
        Files.writeString(cachedPath, "cached", StandardCharsets.UTF_8);

        IOException error =
                assertThrows(
                        IOException.class,
                        () ->
                                cache.fetchHuggingFace(
                                        "unsloth",
                                        "Llama-3.2-1B-Instruct-GGUF",
                                        null,
                                        "Llama-3.2-1B-Instruct-Q8_0.gguf",
                                        true,
                                        true));
        assertTrue(error.getMessage().contains("useCacheOnly=true"));
    }

    @Test
    void fetchPartialMetadata_downloadsAndCaches() throws Exception {
        byte[] ggufBytes = minimalGguf();
        AtomicInteger hits = new AtomicInteger();
        startServer(
                "/model.gguf",
                exchange -> writeBytes(exchange, ggufBytes, 200, hits.incrementAndGet()));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/model.gguf");
        Path target = tempDir.resolve("model.gguf.metadata");

        Path first = cache.fetchPartialMetadata("test", url, target, null, false, false);
        Path second = cache.fetchPartialMetadata("test", url, target, null, false, false);

        assertEquals(first, second);
        assertEquals(1, hits.get(), "second fetch should reuse cache");
        assertTrue(Files.size(first) > 0);
        GGUF parsed = GGUF.read(first);
        assertNotNull(parsed);
        assertEquals("test-value", parsed.getString("test.key"));
    }

    @Test
    void fetchPartialMetadata_forceRefreshRedownloads() throws Exception {
        byte[] ggufBytes = minimalGguf();
        AtomicInteger hits = new AtomicInteger();
        startServer(
                "/refresh.gguf",
                exchange -> writeBytes(exchange, ggufBytes, 200, hits.incrementAndGet()));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/refresh.gguf");
        Path target = tempDir.resolve("refresh.gguf.metadata");

        cache.fetchPartialMetadata("test", url, target, null, false, false);
        cache.fetchPartialMetadata("test", url, target, null, false, true);

        assertEquals(2, hits.get(), "forceRefresh should trigger re-download");
    }

    @Test
    void fetchPartialMetadata_404throws() throws Exception {
        startServer("/missing.gguf", exchange -> writeText(exchange, "not found", 404));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/missing.gguf");
        Path target = tempDir.resolve("missing.gguf.metadata");

        IOException error =
                assertThrows(
                        IOException.class,
                        () -> cache.fetchPartialMetadata("test", url, target, null, false, false));

        assertTrue(error.getMessage().contains("HTTP 404"));
    }

    @Test
    void fetchPartialMetadata_500throws() throws Exception {
        startServer("/error.gguf", exchange -> writeText(exchange, "server error", 500));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/error.gguf");
        Path target = tempDir.resolve("error.gguf.metadata");

        IOException error =
                assertThrows(
                        IOException.class,
                        () -> cache.fetchPartialMetadata("test", url, target, null, false, false));

        assertTrue(error.getMessage().contains("HTTP 500"));
    }

    @Test
    void fetchPartialMetadata_malformedFileThrows() throws Exception {
        byte[] corrupted = minimalGguf();
        corrupted[0] = 0;
        startServer("/corrupt.gguf", exchange -> writeBytes(exchange, corrupted, 200, null));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/corrupt.gguf");
        Path target = tempDir.resolve("corrupt.gguf.metadata");

        IOException error =
                assertThrows(
                        IOException.class,
                        () -> cache.fetchPartialMetadata("test", url, target, null, false, false));

        assertTrue(
                error.getMessage().contains("Failed to parse GGUF metadata"),
                "message should indicate parse failure: " + error.getMessage());
    }

    @Test
    void fetchPartialMetadata_truncatedFileThrows() throws Exception {
        byte[] truncated = Arrays.copyOf(minimalGguf(), 8);
        startServer("/truncated.gguf", exchange -> writeBytes(exchange, truncated, 200, null));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/truncated.gguf");
        Path target = tempDir.resolve("truncated.gguf.metadata");

        assertThrows(
                IOException.class,
                () -> cache.fetchPartialMetadata("test", url, target, null, false, false));
    }

    @Test
    void fetchPartialMetadata_cleansUpPartialOnFailure() throws Exception {
        byte[] corrupted = minimalGguf();
        corrupted[0] = 0;
        startServer("/cleanup.gguf", exchange -> writeBytes(exchange, corrupted, 200, null));

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/cleanup.gguf");
        Path target = tempDir.resolve("cleanup.gguf.metadata");
        Path partial = tempDir.resolve("cleanup.gguf.metadata.partial");

        assertThrows(
                IOException.class,
                () -> cache.fetchPartialMetadata("test", url, target, null, false, false));

        assertFalse(Files.exists(partial), ".partial file should be cleaned up");
        assertFalse(Files.exists(target), ".metadata file should not be created on failure");
    }

    @Test
    void fetchPartialMetadata_forwardsAuthHeader() throws Exception {
        byte[] ggufBytes = minimalGguf();
        startServer(
                "/auth.gguf",
                exchange -> {
                    String auth = exchange.getRequestHeaders().getFirst("Authorization");
                    if (!"Bearer test-token".equals(auth)) {
                        writeText(exchange, "unauthorized", 403);
                        return;
                    }
                    writeBytes(exchange, ggufBytes, 200, null);
                });

        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = serverUrl("/auth.gguf");
        Path target = tempDir.resolve("auth.gguf.metadata");

        Path result = cache.fetchPartialMetadata("test", url, target, "test-token", false, false);

        assertTrue(Files.exists(result));
    }

    @Test
    void fetchPartialMetadata_useCacheOnlyMissThrows() throws Exception {
        GGUFMetadataCache cache = GGUFMetadataCache.create(tempDir);
        String url = "http://localhost:1/nonexistent.gguf";
        Path target = tempDir.resolve("nonexistent.gguf.metadata");

        IOException error =
                assertThrows(
                        IOException.class,
                        () -> cache.fetchPartialMetadata("test", url, target, null, true, false));

        assertTrue(error.getMessage().contains("useCacheOnly=true"));
    }

    private String serverUrl(String path) {
        return "http://localhost:" + server.getAddress().getPort() + path;
    }

    private void startServer(String path, ThrowingHandler handler) throws IOException {
        server = HttpServer.create(new InetSocketAddress(0), 0);
        server.createContext(
                path,
                exchange -> {
                    try {
                        handler.handle(exchange);
                    } finally {
                        exchange.close();
                    }
                });
        server.start();
    }

    private static void writeText(HttpExchange exchange, String text, int status)
            throws IOException {
        byte[] bytes = text.getBytes(StandardCharsets.UTF_8);
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static void writeBytes(
            HttpExchange exchange, byte[] bytes, int status, Integer hitCounter)
            throws IOException {
        exchange.sendResponseHeaders(status, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    private static byte[] minimalGguf() throws IOException {
        GGUF gguf = Builder.newBuilder().setVersion(3).putString("test.key", "test-value").build();
        Path temp = Files.createTempFile("test-", ".gguf");
        Files.delete(temp);
        try {
            GGUF.write(gguf, temp);
            return Files.readAllBytes(temp);
        } finally {
            Files.deleteIfExists(temp);
        }
    }

    @FunctionalInterface
    private interface ThrowingHandler {
        void handle(HttpExchange exchange) throws IOException;
    }
}
