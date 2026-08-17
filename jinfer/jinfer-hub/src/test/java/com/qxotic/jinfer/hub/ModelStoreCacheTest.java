package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The store's own bookkeeping, offline: what {@code cached} reports, what {@code evict} removes,
 * the offline gate, local-path passthrough, {@code resolveAll}, and the instance semantics of
 * {@code standard()} and {@code of()}.
 */
class ModelStoreCacheTest {

    private static final RemoteFile FILE = new RemoteFile("thing-Q8_0.gguf", 7, null);

    @AfterEach
    void restoreAmbientState() {
        System.clearProperty("jinfer.models");
        System.clearProperty("jinfer.offline");
    }

    // ---- cached ----

    @Test
    void cachedRendersRefsForKnownHostsAndPathsForTheRest(@TempDir Path root) throws IOException {
        Path model = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Path foreign = root.resolve("odd-tree/file.bin");
        Files.createDirectories(model.getParent());
        Files.createDirectories(foreign.getParent());
        Files.writeString(model, "weights");
        Files.writeString(foreign, "legacy");
        // the cache's own scaffolding is never a model
        Files.writeString(model.resolveSibling("thing-Q8_0.gguf.part"), "half");
        Files.writeString(root.resolve("CACHEDIR.TAG"), "tag");
        Files.createDirectories(root.resolve(".locks"));
        Files.writeString(root.resolve(".locks/ab.lock"), "");

        List<ModelStore.Cached> mine =
                ModelStore.of(root).cached().stream()
                        .filter(c -> c.ref().contains("acme") || c.ref().contains("odd-tree"))
                        .toList();

        assertEquals(
                List.of(
                        new ModelStore.Cached(foreign.toString(), 6),
                        new ModelStore.Cached("hf.co/acme/thing/thing-Q8_0.gguf", 7)),
                mine,
                "a ref when the first segment is a known host, the absolute path otherwise,"
                        + " sorted, with scaffolding left out");
    }

    @Test
    void cachedOnAMissingRootListsNothingOfOurs(@TempDir Path root) {
        List<ModelStore.Cached> all = ModelStore.of(root.resolve("absent")).cached();
        assertTrue(
                all.stream().noneMatch(c -> c.ref().contains("absent")),
                "a root that does not exist contributes nothing");
    }

    // ---- evict ----

    @Test
    void evictRemovesAFlatCacheEntryExactlyOnce(@TempDir Path root) throws IOException {
        Path model = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(model.getParent());
        Files.writeString(model, "weights");
        ModelStore store = ModelStore.of(root);

        assertTrue(store.evict("hf.co/acme/thing:Q8_0"));
        assertTrue(Files.notExists(model));
        assertFalse(store.evict("hf.co/acme/thing:Q8_0"), "the second evict is a miss");
    }

    @Test
    void evictNeverTouchesALocalPathOrAnAmbiguousCache(@TempDir Path root) throws IOException {
        Path local = Files.writeString(root.resolve("mine.gguf"), "weights");
        assertFalse(ModelStore.of(root).evict(local.toString()));
        assertTrue(Files.exists(local), "a file passed by path is the caller's, not the cache's");

        // two files one quant could mean: no cache-side answer, so no eviction either
        Path dir = root.resolve("hf.co/acme/thing");
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("a-Q8_0.gguf"), "a");
        Files.writeString(dir.resolve("b-Q8_0.gguf"), "b");
        assertFalse(ModelStore.of(root).evict("hf.co/acme/thing:Q8_0"));
        assertTrue(Files.exists(dir.resolve("a-Q8_0.gguf")));
    }

    // ---- the offline gate ----

    @Test
    void offlineRefusesAMissButServesAHit(@TempDir Path root) throws IOException {
        System.setProperty("jinfer.offline", "true");
        ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", FILE));

        var failure =
                assertThrows(
                        IllegalStateException.class, () -> store.resolve("hf.co/acme/thing:Q8_0"));
        assertTrue(failure.getMessage().contains("JINFER_OFFLINE"), failure.getMessage());

        Path planted = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(planted.getParent());
        Files.writeString(planted, "weights");
        assertEquals(planted, store.resolve("hf.co/acme/thing:Q8_0"));
    }

    @Test
    void aPartialDownloadIsAMissEvenWhenItLooksAlmostComplete(@TempDir Path root)
            throws IOException {
        System.setProperty("jinfer.offline", "true");
        Path dir = root.resolve("hf.co/acme/thing");
        Files.createDirectories(dir);
        Files.writeString(dir.resolve("thing-Q8_0.gguf.part"), "nearly all of it");

        var failure =
                assertThrows(
                        IllegalStateException.class,
                        () -> ModelStore.of(root).resolve("hf.co/acme/thing:Q8_0"));
        assertTrue(failure.getMessage().contains("JINFER_OFFLINE"), failure.getMessage());
    }

    // ---- local passthrough ----

    @Test
    void aLocalFilePassesThroughAndADirectoryIsRefusedByName(@TempDir Path root)
            throws IOException {
        Path local = Files.writeString(root.resolve("mine.gguf"), "weights");
        assertEquals(local, ModelStore.of(root).resolve(local.toString()));
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.of(root).resolve(root.toString()));
        assertTrue(failure.getMessage().contains("is a directory"), failure.getMessage());
    }

    // ---- resolveAll ----

    @Test
    void resolveAllPreservesOrderAcrossLocalRemoteAndWarm(@TempDir Path root) throws IOException {
        FakeSource source =
                new FakeSource("fake")
                        .serving("", FILE, new RemoteFile("other-Q4_0.gguf", 4, null))
                        .bytes("fresh");
        Path warm = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(warm.getParent());
        Files.writeString(warm, "warm");
        Path local = Files.writeString(root.resolve("local.gguf"), "local");
        ModelStore store = ModelStore.of(root, source);

        List<Path> paths =
                store.resolveAll(
                        List.of(
                                "hf.co/acme/thing:Q4_0", // a download
                                local.toString(), // a local file
                                "hf.co/acme/thing:Q8_0")); // warm

        assertEquals(root.resolve("hf.co/acme/thing/other-Q4_0.gguf"), paths.get(0));
        assertEquals(local, paths.get(1));
        assertEquals(warm, paths.get(2));
        assertEquals("fresh", Files.readString(paths.get(0)));
        assertEquals("warm", Files.readString(paths.get(2)), "a warm entry is never refetched");
    }

    @Test
    void resolveAllThrowsTheFailureNotAWrapper(@TempDir Path root) {
        FakeSource source = new FakeSource("fake").serving(""); // no files anywhere
        ModelStore store = ModelStore.of(root, source);

        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                store.resolveAll(
                                        List.of("hf.co/acme/thing:Q8_0", "hf.co/acme/other:Q8_0")));
        assertTrue(failure.getMessage().contains("no .gguf files"), failure.getMessage());
    }

    // ---- the instances themselves ----

    @Test
    void standardBuildsFreshFromTheAmbientProperty(@TempDir Path a, @TempDir Path b) {
        System.setProperty("jinfer.models", a.toString());
        assertEquals(a.toAbsolutePath().normalize(), ModelStore.standard().root());
        System.setProperty("jinfer.models", b.toString());
        assertEquals(
                b.toAbsolutePath().normalize(),
                ModelStore.standard().root(),
                "a property set after the last standard() call is honored by the next");
    }

    @Test
    void ofNormalizesTheRoot(@TempDir Path root) {
        Path messy = root.resolve("sub").resolve("..").resolve(".");
        assertEquals(root.toAbsolutePath().normalize(), ModelStore.of(messy).root());
    }
}
