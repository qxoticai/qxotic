package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.*;

import java.io.IOException;
import java.net.URI;
import java.nio.file.FileSystems;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class HubRefreshTest {
    @TempDir Path cache;
    private static final ModelRef REF = ModelRef.parse("hf.co/refresh-test/model:Q8_0");
    private static final String OLD = "a".repeat(40), NEW = "b".repeat(40);

    @Test
    void publishesSnapshotBeforeRefAndPreservesOldSnapshots() throws Exception {
        try (FileServer server = FileServer.start()) {
            RepositorySource source = new HuggingFaceSource(URI.create(server.url("")));
            RemoteFile before = file("before"), after = file("after!");
            server.serve(url(OLD), "before").serve(url(NEW), "after!");
            Path old = Hub.fetchInto(REF, before, OLD, cache, source);
            assertEquals(OLD, Files.readString(refFile()));
            Path fresh = Hub.fetchInto(REF, after, NEW, cache, source);
            assertEquals(NEW, Files.readString(refFile()));
            assertEquals("after!", Files.readString(fresh));
            assertEquals("before", Files.readString(old));
            assertEquals(fresh.getParent(), Hub.snapshot(REF, cache));
            int fetched = server.hits(url(NEW));
            Files.setLastModifiedTime(refFile(), FileTime.fromMillis(1_000));
            FileTime publishedAt = Files.getLastModifiedTime(refFile());
            Hub.fetchInto(REF, after, NEW, cache, source);
            assertEquals(fetched, server.hits(url(NEW)), "unchanged blob reused");
            assertEquals(
                    publishedAt,
                    Files.getLastModifiedTime(refFile()),
                    "checking an unchanged revision must not rewrite its ref");
            Hub.fetchInto(REF, after, NEW, cache, source, true);
            assertEquals(fetched + 1, server.hits(url(NEW)), "same-hash force downloads again");
        }
    }

    @Test
    void failedDownloadAndFailedSnapshotPublicationDoNotAdvanceRef() throws Exception {
        try (FileServer server = FileServer.start()) {
            RepositorySource source = new HuggingFaceSource(URI.create(server.url("")));
            server.serve(url(OLD), "before").serve(url(NEW), "broken");
            Path old = Hub.fetchInto(REF, file("before"), OLD, cache, source);
            assertThrows(
                    IOException.class,
                    () -> Hub.fetchInto(REF, file("after!"), NEW, cache, source));
            assertEquals(OLD, Files.readString(refFile()));
            assertEquals("before", Files.readString(old));
            // A non-empty directory makes atomic publication of the snapshot entry fail.
            Path blocked =
                    cache.resolve(
                            "models--refresh-test--model/snapshots/" + NEW + "/model-Q8_0.gguf");
            Files.createDirectories(blocked);
            Files.writeString(blocked.resolve("keep"), "keep");
            server.serve(url(NEW), "after!");
            assertThrows(
                    IOException.class,
                    () -> Hub.fetchInto(REF, file("after!"), NEW, cache, source));
            assertEquals(OLD, Files.readString(refFile()));
            assertEquals(old.getParent(), Hub.snapshot(REF, cache));
            assertEquals("before", Files.readString(old));
        }
    }

    @Test
    void noSymlinkFilesystemCopiesWithoutConsumingTheSharedBlob() throws Exception {
        try (var zip =
                FileSystems.newFileSystem(
                        URI.create("jar:" + cache.resolve("cache.zip").toUri()),
                        Map.of("create", "true"))) {
            Path blob = zip.getPath("/blobs/content"),
                    first = zip.getPath("/snapshots/one/model.gguf"),
                    second = zip.getPath("/snapshots/two/model.gguf");
            Files.createDirectories(blob.getParent());
            Files.writeString(blob, "weights");
            Hub.link(blob, first);
            Hub.link(blob, second);
            assertEquals("weights", Files.readString(blob));
            assertEquals("weights", Files.readString(first));
            assertEquals("weights", Files.readString(second));
            assertFalse(Files.isSymbolicLink(first));
        }
    }

    @Test
    void repairsASharedBlobAndWrongSnapshotLinkWithoutUnlinkingOthers() throws Exception {
        try (FileServer server = FileServer.start()) {
            RepositorySource source = new HuggingFaceSource(URI.create(server.url("")));
            server.serve(url(OLD), "before").serve(url(NEW), "before");
            Path first = Hub.fetchInto(REF, file("before"), OLD, cache, source);
            Path second = Hub.fetchInto(REF, file("before"), NEW, cache, source);
            Path blob =
                    cache.resolve("models--refresh-test--model/blobs/" + file("before").sha256());
            Files.writeString(blob, "broken");
            Hub.fetchInto(REF, file("before"), NEW, cache, source, true);
            assertEquals("before", Files.readString(first));
            assertEquals("before", Files.readString(second));
            Files.delete(second);
            Files.createSymbolicLink(second, Path.of("../../blobs/missing"));
            Hub.fetchInto(REF, file("before"), NEW, cache, source);
            assertEquals("before", Files.readString(second));
            assertEquals("before", Files.readString(first));
        }
    }

    private Path refFile() {
        return cache.resolve("models--refresh-test--model/refs/main");
    }

    private static String url(String revision) {
        return "/refresh-test/model/resolve/" + revision + "/model-Q8_0.gguf";
    }

    private static RemoteFile file(String bytes) throws Exception {
        return new RemoteFile("model-Q8_0.gguf", bytes.length(), ModelStorePullTest.hash(bytes));
    }
}
