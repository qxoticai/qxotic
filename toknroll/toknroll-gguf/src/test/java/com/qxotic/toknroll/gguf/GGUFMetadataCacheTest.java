package com.qxotic.toknroll.gguf;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class GGUFMetadataCacheTest {

    @TempDir Path tempDir;

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
}
