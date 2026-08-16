package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.net.InetAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * Live against both hosts. {@code ggml-org/stories15M_MOE} exists on HuggingFace AND ModelScope
 * with byte-identical files, which is what makes it the right fixture here: the same ref through
 * two different listing APIs and two different CDNs must produce the same sha256 on disk.
 *
 * <p>Sizes are chosen to cover both transfer paths - Q8_0 (39 MB) goes down one stream, F16 (73 MB)
 * is cut into chunks and fetched in parallel. Tagged {@code integration}, so the default build does
 * not go to the network.
 */
@Tag("integration")
class ModelHubIT {

    private static final String REPO = "hf.co/ggml-org/stories15M_MOE";
    private static final String MS_REPO = "modelscope.cn/ggml-org/stories15M_MOE";

    /**
     * Skips rather than fails when a host is unreachable. These tests assert what the hub does
     * against a live API; they cannot assert anything about a machine with no route to it, and a
     * red suite on a train is a suite people learn to ignore.
     */
    private static void assumeReachable(String host) {
        try {
            InetAddress.getByName(host);
        } catch (UnknownHostException offline) {
            Assumptions.abort(host + " is unreachable: " + offline);
        }
    }

    private static void useCache(Path root) {
        System.setProperty("jinfer.models", root.toString());
    }

    private static String sha256(Path file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (var in = Files.newInputStream(file)) {
            byte[] buffer = new byte[1 << 20];
            for (int n; (n = in.read(buffer)) > 0; ) {
                digest.update(buffer, 0, n);
            }
        }
        return HexFormat.of().formatHex(digest.digest());
    }

    @Test
    void huggingFaceSequentialDownloadThenCacheHit(@TempDir Path root) throws Exception {
        assumeReachable("huggingface.co");
        useCache(root);
        Path file = ModelStore.standard().resolve(REPO + ":Q8_0");
        assertEquals(root.resolve("hf.co/ggml-org/stories15M_MOE/stories15M_MOE-Q8_0.gguf"), file);
        assertEquals(39390272L, Files.size(file));
        String hash = sha256(file);

        // second resolve must be served from disk, and must be the same bytes
        Path again = ModelStore.standard().resolve(REPO + ":Q8_0");
        assertEquals(file, again);
        assertEquals(hash, sha256(again));

        // no scaffolding left behind
        assertTrue(
                Files.notExists(
                        root.resolve("hf.co/ggml-org/stories15M_MOE")
                                .resolve("stories15M_MOE-Q8_0.gguf.part")));
        assertTrue(Files.exists(root.resolve("CACHEDIR.TAG")));
    }

    @Test
    void huggingFaceParallelChunkedDownload(@TempDir Path root) throws Exception {
        assumeReachable("huggingface.co");
        useCache(root);
        Path file =
                ModelStore.standard()
                        .resolve(REPO + ":F16"); // 73 MB: three chunks, fetched concurrently
        assertEquals(73466432L, Files.size(file));
        assertTrue(
                Files.notExists(file.resolveSibling(file.getFileName() + ".map")),
                "the chunk map is deleted once the file is published");
    }

    /**
     * The default-root behavior, driven through the package seams so the test never touches the
     * machine's real {@code ~/.cache/huggingface}: an {@code hf.co} download lands in the shared
     * hub layout - blob named by its sha256, snapshot symlink, {@code refs/main} - and jinfer's own
     * read side (and llama.cpp's, which reads the same layout) finds it as a cache hit.
     */
    @Test
    void writeThroughPopulatesTheSharedHubLayout(@TempDir Path root, @TempDir Path hub)
            throws Exception {
        assumeReachable("huggingface.co");
        useCache(root);
        ModelRef ref = ModelRef.parse(REPO + ":Q8_0");
        String commit = Hub.commit(ref);
        Assertions.assertNotNull(commit, "main resolves to a commit");

        Path file =
                Hub.fetchInto(
                        ref,
                        ModelStore.standard().select(ref, new HuggingFaceSource()),
                        commit,
                        hub);
        assertTrue(file.startsWith(hub));
        assertEquals(39390272L, Files.size(file));
        assertTrue(Files.isSymbolicLink(file), "a snapshot entry is a link into blobs/");
        Path blob = file.getParent().resolve(Files.readSymbolicLink(file)).normalize();
        assertEquals("blobs", blob.getParent().getFileName().toString());
        assertEquals(blob.getFileName().toString(), sha256(file), "the blob is named by its hash");
        assertEquals(
                commit,
                Files.readString(hub.resolve("models--ggml-org--stories15M_MOE/refs/main"))
                        .strip());

        // the read side sees what the write side did: resolve() from here is a cache hit
        assertEquals(file.getParent(), Hub.snapshot(ref, hub));
        assertTrue(
                Hub.cached(hub).stream()
                        .anyMatch(
                                m ->
                                        m.ref()
                                                        .equals(
                                                                "hf.co/ggml-org/stories15M_MOE/stories15M_MOE-Q8_0.gguf")
                                                && m.sizeBytes() == 39390272L));
    }

    @Test
    void modelScopeServesTheSameBytes(@TempDir Path root) throws Exception {
        assumeReachable("modelscope.cn");
        useCache(root);
        Path fromHf = ModelStore.standard().resolve(REPO + ":Q8_0");
        Path fromMs = ModelStore.standard().resolve(MS_REPO + ":Q8_0");
        assertEquals(root.resolve("modelscope.cn/ggml-org/stories15M_MOE"), fromMs.getParent());
        assertEquals(sha256(fromHf), sha256(fromMs));
    }

    @Test
    void aResumeCompletesAPartialDownload(@TempDir Path root) throws Exception {
        useCache(root);
        assumeReachable("huggingface.co");
        Path dir = root.resolve("hf.co/ggml-org/stories15M_MOE");
        Files.createDirectories(dir);
        Path part = dir.resolve("stories15M_MOE-Q8_0.gguf.part");
        // half a file from a previous run, with the real leading bytes
        Path whole = ModelStore.standard().resolve(REPO + ":Q8_0");
        byte[] prefix = new byte[1 << 20];
        try (var in = Files.newInputStream(whole)) {
            in.readNBytes(prefix, 0, prefix.length);
        }
        String expected = sha256(whole);
        Files.delete(whole);
        Files.write(part, prefix);

        Path resumed = ModelStore.standard().resolve(REPO + ":Q8_0");
        assertEquals(expected, sha256(resumed));
    }

    @Test
    void aParallelResumeRefetchesOnlyTheMissingChunks(@TempDir Path root) throws Exception {
        assumeReachable("huggingface.co");
        useCache(root);
        Path whole = ModelStore.standard().resolve(REPO + ":F16"); // 73 MB = three 32 MB chunks
        String expected = sha256(whole);
        long size = Files.size(whole);
        byte[] firstChunk = new byte[32 << 20];
        try (var in = Files.newInputStream(whole)) {
            in.readNBytes(firstChunk, 0, firstChunk.length);
        }
        Files.delete(whole);

        // what an interrupted parallel download leaves: a full-size sparse file with chunk 0
        // written, and a map saying so. Chunks 1 and 2 are holes.
        Path part = whole.resolveSibling(whole.getFileName() + ".part");
        try (var allocated = new RandomAccessFile(part.toFile(), "rw")) {
            allocated.setLength(size);
            allocated.getChannel().write(ByteBuffer.wrap(firstChunk), 0);
        }
        Files.write(part.resolveSibling(part.getFileName() + ".map"), new byte[] {1, 0, 0});
        assertEquals(size - (32L << 20), Fetch.remainingBytes(whole, size), "two chunks to go");

        Path resumed = ModelStore.standard().resolve(REPO + ":F16");
        assertEquals(expected, sha256(resumed), "a resumed file is byte-identical");
    }

    @Test
    void missingQuantNamesWhatTheRepositoryHas(@TempDir Path root) {
        assumeReachable("huggingface.co");
        useCache(root);
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> ModelStore.standard().resolve(REPO + ":Q4_K_M"));
        assertTrue(failure.getMessage().contains("stories15M_MOE-Q8_0.gguf"), failure.getMessage());
    }

    @Test
    void missingRepositoryNamesItsUrl(@TempDir Path root) {
        assumeReachable("huggingface.co");
        useCache(root);
        var failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                ModelStore.standard()
                                        .resolve("hf.co/ggml-org/there-is-no-such-repo-here"));
        assertTrue(failure.getMessage().contains("huggingface.co"), failure.getMessage());
    }

    @Test
    void offlineRefusesInsteadOfDownloading(@TempDir Path root) {
        useCache(root);
        System.setProperty("jinfer.offline", "true");
        try {
            var failure =
                    assertThrows(
                            IllegalStateException.class,
                            () -> ModelStore.standard().resolve(REPO + ":Q8_0"));
            assertTrue(failure.getMessage().contains("JINFER_OFFLINE"), failure.getMessage());
        } finally {
            System.clearProperty("jinfer.offline");
        }
    }

    @Test
    void anExistingPathIsNeverTreatedAsARef(@TempDir Path dir) throws IOException {
        Path local = Files.writeString(dir.resolve("local-model.gguf"), "not really a model");
        assertEquals(local, ModelStore.standard().resolve(local.toString()));
    }
}
