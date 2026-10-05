package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermissions;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import java.util.concurrent.TimeUnit;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

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
        Files.writeString(model.resolveSibling("thing-Q8_0.gguf.part.etag"), "\"v1\"");
        Files.writeString(root.resolve("CACHEDIR.TAG"), "tag");

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

    /** One directory the walk cannot enter costs the listing nothing but that directory. */
    @Test
    void cachedSkipsWhatItCannotReadAndListsTheRest(@TempDir Path root) throws IOException {
        Path model = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Files.createDirectories(model.getParent());
        Files.writeString(model, "weights");
        Path locked = Files.createDirectories(root.resolve("hf.co/other/locked"));
        Assumptions.assumeTrue(
                Files.getFileStore(root).supportsFileAttributeView("posix"), "POSIX permissions");
        Files.setPosixFilePermissions(locked, Set.of());
        Assumptions.assumeFalse(Files.isReadable(locked), "root can read anything");
        try {
            assertEquals(
                    List.of(new ModelStore.Cached("hf.co/acme/thing/thing-Q8_0.gguf", 7)),
                    ModelStore.of(root).cached().stream()
                            .filter(c -> c.ref().contains("acme") || c.ref().contains("other"))
                            .toList());
        } finally {
            Files.setPosixFilePermissions(locked, PosixFilePermissions.fromString("rwxr-xr-x"));
        }
    }

    /**
     * A ref in both caches lists once, as the copy the store serves: its own, whatever the hub's
     * size.
     */
    @Test
    void mergedListsEachRefOnceTheStoresOwnCopyFirst() {
        var own =
                List.of(
                        new ModelStore.Cached("hf.co/acme/thing/thing-Q8_0.gguf", 7),
                        new ModelStore.Cached("/models/local.gguf", 1));
        var hub =
                List.of(
                        new ModelStore.Cached("hf.co/acme/thing/thing-Q8_0.gguf", 9),
                        new ModelStore.Cached("hf.co/other/thing/thing-Q4_0.gguf", 3));
        assertEquals(
                List.of(
                        new ModelStore.Cached("/models/local.gguf", 1),
                        new ModelStore.Cached("hf.co/acme/thing/thing-Q8_0.gguf", 7),
                        new ModelStore.Cached("hf.co/other/thing/thing-Q4_0.gguf", 3)),
                ModelStore.merged(own, hub),
                "one line per ref, sorted, the served copy's size");
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

    @ParameterizedTest
    @CsvSource({
        "1,,JINFER_OFFLINE",
        "true,,JINFER_OFFLINE",
        "ON,,JINFER_OFFLINE",
        "YeS,,JINFER_OFFLINE",
        "0,,online",
        "FALSE,,online",
        "off,,online",
        "No,,online",
        ",,online",
        "'',,online",
        "invalid,,online",
        // the property decides when set, parsed like the variable; the refusal names it
        "off,true,-Djinfer.offline",
        ",YES,-Djinfer.offline",
        "1,false,online",
        "1,off,online",
        "1,invalid,JINFER_OFFLINE"
    })
    void offlineSettingsGateRemoteAccess(
            String value, String property, String expected, @TempDir Path root) throws Exception {
        String classes =
                Path.of(
                                ModelStore.class
                                        .getProtectionDomain()
                                        .getCodeSource()
                                        .getLocation()
                                        .toURI())
                        .toString();
        Path output = root.resolve("stdout.txt"), error = root.resolve("stderr.txt");
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.add("-ea");
        if (property != null) command.add("-Djinfer.offline=" + property);
        command.addAll(
                List.of(
                        "-cp",
                        classes + File.pathSeparator + System.getProperty("java.class.path"),
                        OfflineProbe.class.getName(),
                        root.toString()));
        var builder =
                new ProcessBuilder(command)
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
        for (String variable : List.of("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"))
            builder.environment().remove(variable);
        if (value == null) builder.environment().remove("JINFER_OFFLINE");
        else builder.environment().put("JINFER_OFFLINE", value);
        builder.environment().put("HF_HUB_CACHE", root.resolve("hf").toString());
        Process process = builder.start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "offline probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(error));
            assertEquals(expected, Files.readString(output).strip());
        } finally {
            process.destroyForcibly();
        }
    }

    public static class OfflineProbe {
        public static void main(String[] args) throws Exception {
            var source = new FakeSource("fake").serving("", FILE);
            var store = ModelStore.of(Path.of(args[0]), source);
            String ref = "hf.co/offline-test/model/thing-Q8_0.gguf";
            try {
                assert "weights".equals(Files.readString(store.resolve(ref)));
                assert source.fetched();
                System.out.println("online");
            } catch (IllegalStateException offline) {
                String setting =
                        offline.getMessage().contains("-Djinfer.offline")
                                ? "-Djinfer.offline"
                                : offline.getMessage().contains("JINFER_OFFLINE")
                                        ? "JINFER_OFFLINE"
                                        : null;
                if (setting == null) throw offline;
                assert source.requestedDirs().isEmpty()
                        : "offline must prevent metadata requests too";
                assert !source.fetched();
                System.out.println(setting);
            }
        }
    }

    @Test
    void offlineRefusesAMissButServesAHit(@TempDir Path root) throws IOException {
        System.setProperty("jinfer.offline", "true");
        ModelStore store = ModelStore.of(root, new FakeSource("fake").serving("", FILE));

        var failure =
                assertThrows(
                        IllegalStateException.class, () -> store.resolve("hf.co/acme/thing:Q8_0"));
        assertTrue(failure.getMessage().contains("-Djinfer.offline"), failure.getMessage());

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
        assertTrue(failure.getMessage().contains("-Djinfer.offline"), failure.getMessage());
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
