package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The {@link ModelSource} contract through a fake in-memory source - no network, so these gates run
 * in the default build: ordered fallback, what falls through and what does not, and the offline and
 * read-only stores.
 */
class ModelStoreFallbackTest {

    private static final String REF = "hf.co/acme/thing:Q8_0";
    private static final RemoteFile FILE = new RemoteFile("thing-Q8_0.gguf", 7, null);

    /** A source that answers from memory, or fails the way it was told to. */
    private static final class FakeSource implements ModelSource {
        final String name;
        List<RemoteFile> listing = List.of(FILE);
        Exception failure; // thrown by list, when set
        boolean listed;
        boolean fetched;

        FakeSource(String name) {
            this.name = name;
        }

        FakeSource failing(Exception failure) {
            this.failure = failure;
            return this;
        }

        FakeSource listing(List<RemoteFile> files) {
            this.listing = files;
            return this;
        }

        @Override
        public boolean supports(ModelRef ref) {
            return ref.host().equals("hf.co");
        }

        @Override
        public List<RemoteFile> list(ModelRef ref, String dir) throws IOException {
            listed = true;
            if (failure instanceof IOException io) {
                throw io;
            }
            if (failure instanceof RuntimeException runtime) {
                throw runtime;
            }
            return listing;
        }

        @Override
        public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
            fetched = true;
            Files.createDirectories(into.getParent());
            Files.writeString(into, "weights");
        }

        @Override
        public String toString() {
            return name;
        }
    }

    @Test
    void theFirstServingSourceWins(@TempDir Path root) throws IOException {
        FakeSource first = new FakeSource("first");
        FakeSource second = new FakeSource("second");

        Path file = ModelStore.of(root, first, second).resolve(REF);

        assertEquals(root.resolve("hf.co/acme/thing/thing-Q8_0.gguf"), file);
        assertEquals("weights", Files.readString(file));
        assertTrue(first.fetched);
        assertFalse(second.listed, "a later source is never asked");
    }

    @Test
    void aFailedSourceFallsThroughToTheNext(@TempDir Path root) throws IOException {
        FakeSource down = new FakeSource("down").failing(new IOException("connection refused"));
        FakeSource up = new FakeSource("up");

        Path file = ModelStore.of(root, down, up).resolve(REF);

        assertEquals("weights", Files.readString(file));
        assertTrue(up.fetched, "the next source serves it");
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
        FakeSource up = new FakeSource("up");

        Path file = ModelStore.of(root, missing, up).resolve(REF);

        assertEquals("weights", Files.readString(file));
        assertTrue(up.fetched);
    }

    @Test
    void aSelectionFailureIsTheRefsFaultAndStopsTheSearch(@TempDir Path root) {
        FakeSource empty = new FakeSource("empty").listing(List.of());
        FakeSource later = new FakeSource("later");

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root, empty, later).resolve(REF));
        assertTrue(failure.getMessage().contains("no .gguf files"), failure.getMessage());
        assertFalse(later.listed, "the ref is at fault: asking another source would repeat it");
    }

    @Test
    void aMissEverywhereNamesEverySource(@TempDir Path root) {
        FakeSource a = new FakeSource("a").failing(new IOException("down a"));
        FakeSource b = new FakeSource("b").failing(new IOException("down b"));

        var failure =
                assertThrows(
                        UncheckedIOException.class, () -> ModelStore.of(root, a, b).resolve(REF));
        assertTrue(failure.getMessage().contains("down a"), failure.getMessage());
        assertTrue(failure.getMessage().contains("down b"), failure.getMessage());
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
            ModelStore store = ModelStore.of(root, new FakeSource("fake"));

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
