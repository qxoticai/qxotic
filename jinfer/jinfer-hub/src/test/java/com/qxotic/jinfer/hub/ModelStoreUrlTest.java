package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
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
