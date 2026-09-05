package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The {@link ModelSource} contract through {@link FakeSource} - no network, so these gates run in
 * the default build: ordered fallback, what falls through and what does not, and the offline and
 * read-only stores.
 */
class ModelStoreFallbackTest {

    private static final String REF = "hf.co/acme/thing:Q8_0";
    private static final RemoteFile FILE = new RemoteFile("thing-Q8_0.gguf", 7, null);

    @Test
    void theFirstServingSourceWins(@TempDir Path root) throws IOException {
        FakeSource first = new FakeSource("first").serving("", FILE);
        FakeSource second = new FakeSource("second").serving("", FILE);

        Path file = ModelStore.of(root, first, second).resolve(REF);

        assertEquals(root.resolve("hf.co/acme/thing/thing-Q8_0.gguf"), file);
        assertEquals("weights", Files.readString(file));
        assertTrue(first.fetched());
        assertTrue(second.requestedDirs().isEmpty(), "a later source is never asked");
    }

    @Test
    void aFailedSourceFallsThroughToTheNext(@TempDir Path root) throws IOException {
        FakeSource down = new FakeSource("down").failing(new IOException("connection refused"));
        FakeSource up = new FakeSource("up").serving("", FILE);

        Path file = ModelStore.of(root, down, up).resolve(REF);

        assertEquals("weights", Files.readString(file));
        assertTrue(up.fetched(), "the next source serves it");
    }

    @Test
    void aRepositoryThatAnsweredNoFallsThroughToo(@TempDir Path root) throws IOException {
        // a 404 surfaces as an IllegalArgumentException CAUSED BY the status: the source answered,
        // and its answer was "no" - another source may know the repository
        FakeSource missing =
                new FakeSource("missing")
                        .failing(
                                new IllegalArgumentException(
                                        "no repository acme/thing on hf.co",
                                        new Fetch.HttpStatusException(404, "https://x", "")));
        FakeSource up = new FakeSource("up").serving("", FILE);

        Path file = ModelStore.of(root, missing, up).resolve(REF);

        assertEquals("weights", Files.readString(file));
        assertTrue(up.fetched());
    }

    @Test
    void aSelectionFailureIsTheRefsFaultAndStopsTheSearch(@TempDir Path root) {
        FakeSource empty = new FakeSource("empty").serving("");
        FakeSource later = new FakeSource("later").serving("", FILE);

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, empty, later).resolve(REF));
        assertTrue(failure.getMessage().contains("no .gguf files"), failure.getMessage());
        assertTrue(
                later.requestedDirs().isEmpty(),
                "the ref is at fault: asking another source would repeat it");
    }

    @Test
    void aMissEverywhereThrowsTheLastSourcesOwnAnswer(@TempDir Path root) {
        FakeSource a = new FakeSource("a").failing(new IOException("down a"));
        FakeSource b = new FakeSource("b").failing(new IOException("down b"));

        var failure =
                assertThrows(
                        UncheckedIOException.class, () -> ModelStore.of(root, a, b).resolve(REF));
        assertTrue(failure.getMessage().contains("down b"), failure.getMessage());
        assertFalse(
                failure.getMessage().contains("down a"),
                "the earlier failure is a WARNING in the log, not noise in the message");
    }

    @Test
    void aDeniedDownloadTeachesTheSameAsADeniedListing(@TempDir Path root) {
        // Hugging Face lists a gated repository but answers the file with 401: the user must see
        // the licence/token remedy, not an HTTP status wrapped in a resolve failure
        FakeSource gated =
                new FakeSource("gated")
                        .serving("", new RemoteFile("model-q8_0.gguf", 3, null))
                        .fetchFailing(new Fetch.HttpStatusException(401, "https://x", ""));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, gated).resolve(REF));
        assertTrue(failure.getMessage().contains("gated or private"), failure.getMessage());
        assertTrue(failure.getMessage().contains("does not exist"), failure.getMessage());
        assertTrue(failure.getMessage().contains("HF_TOKEN"), failure.getMessage());
    }

    @Test
    void aSingleSourcesNoIsDeliveredAsItself(@TempDir Path root) {
        // no fallback is possible, so the teaching message must arrive unwrapped: "gated or
        // private, accept the licence, set the token" - not an aggregate of one
        FakeSource missing =
                new FakeSource("missing")
                        .failing(
                                new IllegalArgumentException(
                                        "acme/thing is gated or private",
                                        new Fetch.HttpStatusException(403, "https://x", "")));

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, missing).resolve(REF));
        assertTrue(failure.getMessage().contains("gated or private"), failure.getMessage());
    }

    @Test
    void anOfflineStoreServesCacheHitsOnly(@TempDir Path root) throws IOException {
        ModelStore offline = ModelStore.of(root); // no sources: the offline store, without a flag

        var failure = assertThrows(UncheckedIOException.class, () -> offline.resolve(REF));
        assertTrue(failure.getMessage().contains("no source"), failure.getMessage());

        Path planted = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(planted.getParent());
        Files.writeString(planted, "weights");
        assertEquals(planted, offline.resolve(REF), "a cache hit never needed a source");
    }

    @Test
    void aReadOnlyRootServesHitsAndRefusesMisses(@TempDir Path root) throws IOException {
        Path planted = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(planted.getParent());
        Files.writeString(planted, "weights");
        // the WHOLE tree, not just the root: the store checks the nearest existing ancestor
        try (var walk = Files.walk(root)) {
            walk.filter(Files::isDirectory).forEach(d -> d.toFile().setWritable(false, false));
        }
        Assumptions.assumeTrue(
                !Files.isWritable(root), "this filesystem will not make a directory read-only");
        try {
            ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", FILE));

            assertEquals(planted, store.resolve(REF), "a hit on a read-only root still resolves");
            var failure =
                    assertThrows(
                            UncheckedIOException.class,
                            () -> store.resolve("hf.co/acme/other:Q8_0"));
            assertTrue(failure.getMessage().contains("not writable"), failure.getMessage());
        } finally {
            try (var walk = Files.walk(root)) {
                walk.filter(Files::isDirectory).forEach(d -> d.toFile().setWritable(true, false));
            }
        }
    }
}
