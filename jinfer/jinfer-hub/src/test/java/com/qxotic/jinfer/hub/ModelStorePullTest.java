package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ModelStorePullTest {
    @TempDir Path root;
    private static final String REF = "hf.co/pull-test/model:Q8_0";

    static String hash(String text) throws Exception {
        return HexFormat.of()
                .formatHex(
                        MessageDigest.getInstance("SHA-256")
                                .digest(text.getBytes(StandardCharsets.UTF_8)));
    }

    @Test
    void useUpdateAndRepairHaveDifferentCachePolicies() throws Exception {
        AtomicInteger lists = new AtomicInteger(), fetches = new AtomicInteger();
        String[] remote = {"before"};
        ModelSource source =
                new ModelSource() {
                    public boolean supports(ModelRef ref) {
                        return true;
                    }

                    public List<RemoteFile> list(ModelRef ref, String dir) {
                        lists.incrementAndGet();
                        try {
                            return List.of(
                                    new RemoteFile(
                                            "model-Q8_0.gguf",
                                            remote[0].length(),
                                            hash(remote[0])));
                        } catch (Exception e) {
                            throw new AssertionError(e);
                        }
                    }

                    public void fetch(ModelRef ref, RemoteFile file, Path into) throws IOException {
                        fetches.incrementAndGet();
                        Files.createDirectories(into.getParent());
                        Files.writeString(into, remote[0]);
                    }
                };
        ModelStore store = ModelStore.of(root, source);
        Path model = store.resolve(REF);
        assertEquals(model, store.resolve(REF));
        assertEquals(List.of(model), store.resolveAll(List.of(REF)));
        assertEquals(1, lists.get());
        assertEquals(1, fetches.get());
        assertEquals(List.of(model), store.pullAll(List.of(REF), false));
        assertEquals(2, lists.get());
        assertEquals(1, fetches.get(), "unchanged content costs metadata, not a payload");
        remote[0] = "after!"; // same byte count is not the same model
        store.pullAll(List.of(REF), false);
        assertEquals(2, fetches.get());
        assertEquals("after!", Files.readString(store.resolve(REF)));
        store.pullAll(List.of(REF), true);
        assertEquals(3, fetches.get(), "force bypasses even a matching checksum");
        Files.writeString(model, "broken");
        store.pullAll(List.of(REF), false);
        assertEquals(4, fetches.get());
        assertEquals("after!", Files.readString(model));
    }

    @Test
    void failuresAndBadCustomSourcesCannotDamageThePublishedFile() throws Exception {
        Path dest = root.resolve("hf.co/pull-test/model/model-Q8_0.gguf");
        Files.createDirectories(dest.getParent());
        Files.writeString(dest, "before");
        var expected = new RemoteFile("model-Q8_0.gguf", 6, hash("after!"));
        FakeSource source = new FakeSource("test").serving("", expected).bytes("broken");
        ModelStore store = ModelStore.of(root, source);
        assertThrows(UncheckedIOException.class, () -> store.pullAll(List.of(REF), true));
        assertEquals(
                "before", Files.readString(dest), "custom source's bad checksum was published");
        source.fetchFailing(new IOException("network failed"));
        assertThrows(UncheckedIOException.class, () -> store.pullAll(List.of(REF), false));
        assertEquals("before", Files.readString(dest));
        source.failing(new IOException("metadata failed"));
        assertThrows(UncheckedIOException.class, () -> store.pullAll(List.of(REF), true));
        assertEquals(dest, store.resolve(REF));
        assertEquals("before", Files.readString(dest));
    }

    @Test
    void offlinePullOnlyServesAnExactImmutableFileOrLocalPath() throws Exception {
        String pinned = "hf.co/pull-test/model@" + "a".repeat(40) + "/model-Q8_0.gguf";
        ModelStore store = ModelStore.of(root);
        Path immutable = store.pathOf(ModelRef.parse(pinned), "model-Q8_0.gguf");
        Files.createDirectories(immutable.getParent());
        Files.writeString(immutable, "weights");
        String old = System.getProperty("jinfer.offline");
        System.setProperty("jinfer.offline", "true");
        try {
            assertEquals(List.of(immutable), store.pullAll(List.of(pinned), false));
            assertEquals(List.of(immutable), store.pullAll(List.of(immutable.toString()), true));
            var force =
                    assertThrows(
                            IllegalStateException.class,
                            () -> store.pullAll(List.of(pinned), true));
            assertTrue(force.getMessage().contains("cannot refresh"), force.getMessage());
            assertFalse(force.getMessage().contains("not cached"), force.getMessage());
            assertThrows(IllegalStateException.class, () -> store.pullAll(List.of(REF), false));
            assertEquals("weights", Files.readString(immutable));
        } finally {
            if (old == null) System.clearProperty("jinfer.offline");
            else System.setProperty("jinfer.offline", old);
        }
    }

    @Test
    void renamedOrAmbiguousShorthandCannotReportFalseSuccess() throws Exception {
        Path old = root.resolve("hf.co/pull-test/model/old-Q8_0.gguf");
        Files.createDirectories(old.getParent());
        Files.writeString(old, "before");
        FakeSource source =
                new FakeSource("test")
                        .serving("", new RemoteFile("new-Q8_0.gguf", 6, hash("after!")))
                        .bytes("after!");
        ModelStore store = ModelStore.of(root, source);
        var failure =
                assertThrows(
                        IllegalArgumentException.class, () -> store.pullAll(List.of(REF), false));
        assertTrue(failure.getMessage().contains("explicitly"), failure.getMessage());
        assertFalse(source.fetched());
        assertEquals("before", Files.readString(old));
        String exact = "hf.co/pull-test/model/new-Q8_0.gguf";
        Path fresh = store.pullAll(List.of(exact), false).getFirst();
        assertEquals("after!", Files.readString(store.resolve(exact)));
        assertEquals("new-Q8_0.gguf", fresh.getFileName().toString());
    }

    @Test
    void defaultQuantCanBePulledAlongsideAnotherCachedQuant() throws Exception {
        String ref = "hf.co/pull-test/model";
        Path other = root.resolve("hf.co/pull-test/model/model-Q8_0.gguf");
        Files.createDirectories(other.getParent());
        Files.writeString(other, "before");
        FakeSource source =
                new FakeSource("test")
                        .serving(
                                "",
                                new RemoteFile("model-Q4_K_M.gguf", 6, hash("after!")),
                                new RemoteFile("model-Q8_0.gguf", 6, hash("before")))
                        .bytes("after!");
        ModelStore store = ModelStore.of(root, source);
        Path fresh = store.pullAll(List.of(ref), false).getFirst();
        assertEquals("model-Q4_K_M.gguf", fresh.getFileName().toString());
        assertEquals(fresh, store.resolve(ref));
        assertEquals("after!", Files.readString(fresh));
        assertEquals("before", Files.readString(other));
    }

    @Test
    void soleNonDefaultRemoteFileMustStillWinTheNextLookup() throws Exception {
        String ref = "hf.co/pull-test/model";
        FakeSource source =
                new FakeSource("test")
                        .serving("", new RemoteFile("model-BF16.gguf", 6, hash("after!")))
                        .bytes("after!");
        ModelStore store = ModelStore.of(root, source);
        Path fresh = store.pullAll(List.of(ref), false).getFirst();
        assertEquals(fresh, store.resolve(ref));
        Files.writeString(fresh.resolveSibling("model-Q8_0.gguf"), "before");
        assertThrows(IllegalArgumentException.class, () -> store.pullAll(List.of(ref), false));
        assertEquals("after!", Files.readString(fresh));
    }

    @Test
    void noChecksumDoesNotTurnEqualSizeIntoACacheHit() throws Exception {
        FakeSource source =
                new FakeSource("test")
                        .serving("", new RemoteFile("model-Q8_0.gguf", 6, null))
                        .bytes("before");
        ModelStore store = ModelStore.of(root, source);
        store.resolve(REF);
        source.bytes("after!");
        Path fresh = store.pullAll(List.of(REF), false).getFirst();
        assertEquals("after!", Files.readString(fresh));
    }

    @Test
    void explicitRootPinsRemoteSelectionButKeepsTheOriginalLookupPath() throws Exception {
        String commit = "c".repeat(40);
        try (FileServer server = FileServer.start()) {
            server.serve(
                    "/api/models/pull-test/model/refs",
                    "{\"branches\":[{\"name\":\"main\",\"targetCommit\":\"" + commit + "\"}]}");
            server.serve(
                    "/api/models/pull-test/model/tree/" + commit,
                    "[{\"type\":\"file\",\"path\":\"model-Q8_0.gguf\",\"size\":6,\"lfs\":{\"oid\":\""
                            + hash("after!")
                            + "\"}}]");
            server.serve("/pull-test/model/resolve/" + commit + "/model-Q8_0.gguf", "after!");
            // Branch URLs deliberately do not serve the file: selecting or fetching main fails.
            ModelStore store =
                    ModelStore.of(root, new HuggingFaceSource(URI.create(server.url(""))));
            Path dest = root.resolve("hf.co/pull-test/model/model-Q8_0.gguf");
            Files.createDirectories(dest.getParent());
            Files.writeString(dest, "before");
            assertEquals(List.of(dest), store.pullAll(List.of(REF), false));
            assertEquals("after!", Files.readString(store.resolve(REF)));
            assertFalse(Files.exists(root.resolve("hf.co/pull-test/model@" + commit)));
            int requests = server.hits("/pull-test/model/resolve/" + commit + "/model-Q8_0.gguf");
            store.pullAll(List.of(REF), false);
            assertEquals(
                    requests,
                    server.hits("/pull-test/model/resolve/" + commit + "/model-Q8_0.gguf"));
            store.pullAll(List.of(REF), true);
            assertEquals(
                    requests + 1,
                    server.hits("/pull-test/model/resolve/" + commit + "/model-Q8_0.gguf"));
        }
    }

    @Test
    void oneFailedPullNeverPreEvictsAnotherInput() throws Exception {
        ModelStore store =
                ModelStore.of(root, new FakeSource("down").failing(new IOException("offline")));
        for (String repo : List.of("one", "two")) {
            Path file = root.resolve("hf.co/pull-test/" + repo + "/model-Q8_0.gguf");
            Files.createDirectories(file.getParent());
            Files.writeString(file, repo);
        }
        assertThrows(
                UncheckedIOException.class,
                () ->
                        store.pullAll(
                                List.of("hf.co/pull-test/one:Q8_0", "hf.co/pull-test/two:Q8_0"),
                                true));
        for (String repo : List.of("one", "two"))
            assertEquals(
                    repo, Files.readString(store.resolve("hf.co/pull-test/" + repo + ":Q8_0")));
    }
}
