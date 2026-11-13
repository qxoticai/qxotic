package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.File;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
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
 * The store's own bookkeeping, offline: what {@code cached} reports, what {@code remove} deletes,
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

    // ---- remove ----

    @Test
    void removeDeletesAFlatCacheEntryExactlyOnce(@TempDir Path root) throws IOException {
        Path model = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Path other = root.resolve("hf.co/acme/other/other-Q8_0.gguf");
        Files.createDirectories(model.getParent());
        Files.createDirectories(other.getParent());
        Files.writeString(model, "weights");
        Files.writeString(other, "weights");
        ModelStore store = ModelStore.of(root);

        assertTrue(store.remove("hf.co/acme/thing:Q8_0"));
        assertTrue(Files.notExists(model.getParent()), "the emptied repository folder goes");
        assertTrue(Files.exists(other), "its neighbours stay");
        assertFalse(store.remove("hf.co/acme/thing:Q8_0"), "the second remove is a miss");
        assertTrue(store.remove("hf.co/acme/other:Q8_0"));
        assertTrue(Files.exists(root), "the root itself is never removed");
        assertTrue(Files.notExists(root.resolve("hf.co")));
    }

    @Test
    void removeDeletesTheFileFindReturnsOrAFileInsideTheRoot(@TempDir Path root)
            throws IOException {
        Path q4 = root.resolve("hf.co/acme/thing/thing-Q4_K_M.gguf");
        Path q8 = root.resolve("hf.co/acme/thing/thing-Q8_0.gguf");
        Path other = root.resolve("hf.co/acme/other/other-Q8_0.gguf");
        Path loose = root.resolve("loose/model.gguf");
        for (Path file : List.of(q4, q8, other, loose)) {
            Files.createDirectories(file.getParent());
            Files.writeString(file, "weights");
        }
        ModelStore store = ModelStore.of(root);

        assertTrue(store.remove("acme/thing"), "the default quant, as find returns it");
        assertTrue(Files.notExists(q4));
        assertTrue(Files.exists(q8), "one file, never the whole repository");
        assertTrue(Files.exists(other));
        assertTrue(store.remove(loose.toString()), "list prints this path; remove takes it");
        assertTrue(Files.notExists(root.resolve("loose")));
    }

    @Test
    void removeNeverTouchesALocalPathOrAnAmbiguousCache(@TempDir Path dir) throws IOException {
        Path root = dir.resolve("cache");
        Path local = Files.writeString(dir.resolve("mine.gguf"), "weights");
        assertFalse(ModelStore.of(root).remove(local.toString()));
        assertTrue(
                Files.exists(local), "a file outside the cache is the caller's, not the cache's");

        // two files one quant could mean: no cache-side answer, so no removal either
        Path repo = root.resolve("hf.co/acme/thing");
        Files.createDirectories(repo);
        Files.writeString(repo.resolve("a-Q8_0.gguf"), "a");
        Files.writeString(repo.resolve("b-Q8_0.gguf"), "b");
        assertFalse(ModelStore.of(root).remove("hf.co/acme/thing:Q8_0"));
        assertTrue(Files.exists(repo.resolve("a-Q8_0.gguf")));
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
        Path output = root.resolve("stdout.txt"), error = root.resolve("stderr.txt");
        var builder =
                probe(
                                OfflineProbe.class,
                                property == null
                                        ? List.of()
                                        : List.of("-Djinfer.offline=" + property),
                                root.toString())
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
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

    /**
     * A JVM of this classpath running {@code main}, with none of the options an environment adds.
     */
    private static ProcessBuilder probe(Class<?> main, List<String> flags, String... args) {
        // the module path holds this module and the ones it reads; a plain classpath runs them all
        String path = System.getProperty("java.class.path");
        String modules = System.getProperty("jdk.module.path");
        if (modules != null) path = modules + File.pathSeparator + path;
        List<String> command = new ArrayList<>();
        command.add(Path.of(System.getProperty("java.home"), "bin", "java").toString());
        command.add("-ea");
        command.addAll(flags);
        command.add("-cp");
        command.add(path);
        command.add(main.getName());
        command.addAll(List.of(args));
        var builder = new ProcessBuilder(command);
        for (String variable : List.of("JAVA_TOOL_OPTIONS", "JDK_JAVA_OPTIONS", "_JAVA_OPTIONS"))
            builder.environment().remove(variable);
        return builder;
    }

    @Test
    void aContainersHomeIsFollowedAndAHubCacheItCannotWriteIsOnlyRead(@TempDir Path root)
            throws Exception {
        // the JDK reads user.home from the passwd file, so a container's HOME went unseen; and a
        // miss beside a hub cache this user cannot write failed instead of landing in the store
        Assumptions.assumeFalse(File.separatorChar == '\\', "POSIX homes and permissions");
        Path home = Files.createDirectories(root.resolve("home"));
        Path readOnly = Files.createDirectories(root.resolve("hf"));
        Files.setPosixFilePermissions(readOnly, PosixFilePermissions.fromString("r-xr-xr-x"));
        Path output = root.resolve("stdout.txt"), error = root.resolve("stderr.txt");
        var builder =
                probe(HomeProbe.class, List.of())
                        .redirectOutput(output.toFile())
                        .redirectError(error.toFile());
        for (String variable :
                List.of("XDG_CACHE_HOME", "JINFER_MODELS", "JINFER_OFFLINE", "HF_HOME"))
            builder.environment().remove(variable);
        builder.environment().put("HOME", home.toString());
        builder.environment().put("HF_HUB_CACHE", readOnly.toString());
        Process process = builder.start();
        try {
            assertTrue(process.waitFor(20, TimeUnit.SECONDS), "home probe timed out");
            assertEquals(0, process.exitValue(), Files.readString(error));
            List<String> lines = Files.readAllLines(output);
            assertTrue(Path.of(lines.get(0)).startsWith(home), "the store's root: " + lines);
            assertTrue(
                    Path.of(lines.get(1)).startsWith(lines.get(0)), "the miss, fetched: " + lines);
        } finally {
            process.destroyForcibly();
            Files.setPosixFilePermissions(readOnly, PosixFilePermissions.fromString("rwxr-xr-x"));
        }
    }

    @Test
    void aFailureReadsInWordsWithNoJavaClassName(@TempDir Path root) throws IOException {
        // the messages ended in the exception itself: its class name, and a refused file's path
        // alone
        Assumptions.assumeFalse(File.separatorChar == '\\', "POSIX permissions");
        var store =
                ModelStore.of(
                        root,
                        new FakeSource("down").failing(new IOException("connection refused")));
        var down =
                assertThrows(
                        UncheckedIOException.class, () -> store.resolve("hf.co/words/model:Q8_0"));
        assertEquals(
                "could not resolve hf.co/words/model:Q8_0: connection refused", down.getMessage());
        Path hub = Files.createDirectories(root.resolve("hf"));
        Files.setPosixFilePermissions(hub, PosixFilePermissions.fromString("---------"));
        try {
            var refused = assertThrows(UncheckedIOException.class, () -> Hub.cached(hub));
            assertTrue(
                    refused.getMessage().endsWith(hub + ": permission denied"),
                    refused.getMessage());
            assertFalse(refused.getMessage().contains("java."), refused.getMessage());
        } finally {
            Files.setPosixFilePermissions(hub, PosixFilePermissions.fromString("rwxr-xr-x"));
        }
    }

    /**
     * A hub miss at the platform's default root, served over HTTP so it takes the hub-cache path.
     */
    public static class HomeProbe {
        public static void main(String[] args) throws Exception {
            Path root = ModelStore.ambientRoot();
            System.out.println(root);
            if (!root.startsWith(System.getenv("HOME"))) return; // never fetch into a real cache
            String commit = "c".repeat(40);
            try (FileServer server = FileServer.start()) {
                server.serve(
                        "/api/models/home-test/model/refs",
                        "{\"branches\":[{\"name\":\"main\",\"targetCommit\":\"" + commit + "\"}]}");
                server.serve(
                        "/api/models/home-test/model/tree/" + commit,
                        "[{\"type\":\"file\",\"path\":\"model-Q8_0.gguf\",\"size\":7,\"lfs\":{\"oid\":\""
                                + ModelStorePullTest.hash("weights")
                                + "\"}}]");
                server.serve("/home-test/model/resolve/" + commit + "/model-Q8_0.gguf", "weights");
                var store = ModelStore.of(root, new HuggingFaceSource(URI.create(server.url(""))));
                System.out.println(store.resolve("hf.co/home-test/model:Q8_0"));
            }
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
            } catch (ModelNotFoundException offline) {
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
                        ModelNotFoundException.class, () -> store.resolve("hf.co/acme/thing:Q8_0"));
        assertTrue(failure.getMessage().contains("-Djinfer.offline"), failure.getMessage());
        assertTrue(
                failure.getMessage().contains(root.resolve("hf.co/acme/thing") + " or "),
                failure.getMessage());
        assertTrue(failure.getMessage().contains("models--acme--thing"), failure.getMessage());

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
                        ModelNotFoundException.class,
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
