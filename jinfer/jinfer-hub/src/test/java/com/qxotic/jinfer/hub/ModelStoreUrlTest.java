package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Plain-URL resolution against a {@link FileServer}: the cache mapping, the cache hit, the query
 * handling, and the range-blind server downgrade - everything {@code resolve} does for a URL that
 * is not a repository ref, without leaving the machine.
 */
class ModelStoreUrlTest {

    @Test
    void aRepositoryPageUrlIsItsRef() {
        assertEquals(
                "hf.co/LiquidAI/LFM2.5-350M-GGUF",
                ModelStore.repositoryRef(
                        URI.create("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF")));
        assertEquals(
                "hf.co/LiquidAI/LFM2.5-350M-GGUF",
                ModelStore.repositoryRef(
                        URI.create("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/tree/main")));
        assertEquals(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf",
                ModelStore.repositoryRef(
                        URI.create(
                                "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/mmproj-F32.gguf?download=true")));
        assertEquals(
                "hf.co/unsloth/gemma-4-E2B-it-GGUF@v2/sub/mtp.gguf",
                ModelStore.repositoryRef(
                        URI.create(
                                "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/v2/sub/mtp.gguf")));
        assertEquals(
                "modelscope.cn/Qwen/Qwen3-0.6B-GGUF",
                ModelStore.repositoryRef(
                        URI.create("https://www.modelscope.cn/models/Qwen/Qwen3-0.6B-GGUF")));
        assertNull(
                ModelStore.repositoryRef(
                        URI.create("https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/discussions")),
                "a page that is not the model stays a plain URL");
        assertNull(ModelStore.repositoryRef(URI.create("https://example.org/models/x.gguf")));
        assertNull(ModelStore.repositoryRef(URI.create("https://huggingface.co/")));
    }

    @Test
    void aWebPageIsNeverAModelFile(@TempDir Path root) throws IOException {
        try (FileServer server =
                FileServer.start()
                        .serve(
                                "/models/page.gguf",
                                "\n<!DOCTYPE html><html><body>login</body></html>")) {
            ModelStore store = ModelStore.of(root);
            String url = server.url("/models/page.gguf");

            IllegalArgumentException e =
                    assertThrows(IllegalArgumentException.class, () -> store.resolve(url));

            assertTrue(e.getMessage().contains("web page"), e.getMessage());
            assertTrue(e.getMessage().contains("owner/repo"), e.getMessage());
            assertFalse(
                    Files.exists(root.resolve("127.0.0.1/models/page.gguf")),
                    "the page does not stay in the cache");
        }
    }

    @Test
    void aPlainUrlDownloadsAndCachesByHostAndPath(@TempDir Path root) throws IOException {
        try (FileServer server = FileServer.start().serve("/models/x.gguf", "weights")) {
            ModelStore store = ModelStore.of(root);
            String url = server.url("/models/x.gguf");

            Path file = store.resolve(url);

            assertEquals(root.resolve("127.0.0.1/models/x.gguf"), file);
            assertEquals("weights", Files.readString(file));
            int hits = server.hits("/models/x.gguf");
            assertEquals(file, store.resolve(url), "the second resolve is a cache hit");
            assertEquals(hits, server.hits("/models/x.gguf"), "and costs no request");
        }
    }

    @Test
    void theQueryReachesTheServerButNotTheCachePath(@TempDir Path root) throws IOException {
        try (FileServer server = FileServer.start().serve("/models/signed.gguf", "weights")) {
            Path file =
                    ModelStore.of(root)
                            .resolve(server.url("/models/signed.gguf") + "?sig=abc&expires=1");

            assertEquals("sig=abc&expires=1", server.lastQuery("/models/signed.gguf"));
            assertEquals(root.resolve("127.0.0.1/models/signed.gguf"), file);
        }
    }

    @Test
    void aServerThatIgnoresRangeStillServes(@TempDir Path root) throws IOException {
        try (FileServer server =
                FileServer.start()
                        .serve("/models/plain.gguf", "weights")
                        .ignoringRange("/models/plain.gguf")) {
            Path file = ModelStore.of(root).resolve(server.url("/models/plain.gguf"));

            assertEquals("weights", Files.readString(file));
        }
    }

    @Test
    void aUrlThatWouldEscapeTheCacheIsRefusedBeforeAnyRequest(@TempDir Path root)
            throws IOException {
        try (FileServer server = FileServer.start().serve("/evil.gguf", "weights")) {
            String escape = server.url("/models/../evil.gguf");
            var failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> ModelStore.of(root).resolve(escape));
            assertTrue(failure.getMessage().contains("escape the cache"), failure.getMessage());
            assertEquals(0, server.hits("/evil.gguf"), "the request never left the machine");
        }
    }
}
