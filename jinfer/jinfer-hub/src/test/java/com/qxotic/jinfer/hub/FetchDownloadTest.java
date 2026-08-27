package com.qxotic.jinfer.hub;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

/**
 * The download's own guarantees against a {@link FileServer}: sha256 enforcement, the resume
 * contract, and what is (not) left behind on failure. These are the bytes-on-disk properties every
 * {@link ModelSource} built on {@code Fetch.download} inherits.
 */
class FetchDownloadTest {

    private static final String PAYLOAD =
            "weights, but long enough to have a middle worth resuming from";

    @AfterEach
    void pathLocksAreReleased() throws ReflectiveOperationException {
        Field registry = Fetch.class.getDeclaredField("LOCK_REGISTRY");
        registry.setAccessible(true);
        assertTrue(((Map<?, ?>) registry.get(null)).isEmpty());
    }

    private static String sha256(String value) {
        try {
            return HexFormat.of()
                    .formatHex(
                            MessageDigest.getInstance("SHA-256")
                                    .digest(
                                            value.getBytes(
                                                    java.nio.charset.StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException e) {
            throw new AssertionError(e);
        }
    }

    @Test
    void aMatchingSha256IsVerifiedSilently(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
        }
    }

    @Test
    void aServerIgnoringRangeStillDownloadsCorrectlyAboveTheParallelFloor(@TempDir Path dir)
            throws IOException {
        // every chunk beyond the first used to accept a 200 (the whole file) and write it at its
        // own offset: a corrupt file, published silently when no sha256 was known
        byte[] big = new byte[(int) Fetch.PARALLEL_FLOOR + 1];
        new java.util.Random(11).nextBytes(big);
        try (FileServer server =
                FileServer.start().serve("/big.bin", big).ignoringRange("/big.bin")) {
            Path dest = dir.resolve("big.bin");
            Fetch.download(server.url("/big.bin"), dest, big.length, null, Map.of());
            assertArrayEquals(big, Files.readAllBytes(dest));
        }
    }

    @Test
    void aParallelResumeRestartsWhenTheRemoteChanged(@TempDir Path dir) throws IOException {
        // a chunk map trusted on sizes alone kept a stale chunk of a republished file of the
        // same size; the parallel path now sends the first response's validator as If-Range,
        // and a changed remote (200 to a ranged request) restarts the transfer from scratch
        byte[] big = new byte[(int) Fetch.PARALLEL_FLOOR + 1];
        new java.util.Random(12).nextBytes(big);
        try (FileServer server =
                FileServer.start().serve("/big.bin", big).etag("/big.bin", "\"v2\"")) {
            Path dest = dir.resolve("big.bin");
            Path part = dest.resolveSibling("big.bin.part");
            byte[] stale = new byte[big.length]; // "chunk 0 done" of the previous file
            Files.write(part, stale);
            Files.write(dest.resolveSibling("big.bin.part.map"), new byte[] {1, 0, 0});
            Files.writeString(dest.resolveSibling("big.bin.part.etag"), "\"v1\"");

            Fetch.download(server.url("/big.bin"), dest, big.length, null, Map.of());

            assertEquals("\"v1\"", server.lastHeader("/big.bin", "If-Range"));
            assertArrayEquals(big, Files.readAllBytes(dest));
            assertTrue(!Files.exists(dest.resolveSibling("big.bin.part.etag")), "cleaned up");
        }
    }

    @Test
    void aPartAlreadyAtFullSizeIsNotResumed(@TempDir Path dir) throws IOException {
        // a crash between the last byte and the rename left a full-size .part; resuming it asked
        // for a range past the end, a 416 on every attempt
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            Files.writeString(dest.resolveSibling("m.gguf.part"), PAYLOAD);

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
        }
    }

    @Test
    void aResumeCarriesTheFirstResponsesValidator(@TempDir Path dir) throws IOException {
        // If-Range with the validator the first response gave: a remote that changed answers
        // the whole file, never a tail of a different file appended to the stale prefix
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            Path part = dest.resolveSibling("m.gguf.part");
            Files.writeString(part, PAYLOAD.substring(0, PAYLOAD.length() / 2));
            Files.writeString(dest.resolveSibling("m.gguf.part.etag"), "\"v1\"");

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals("\"v1\"", server.lastHeader("/m.gguf", "If-Range"));
            assertEquals(PAYLOAD, Files.readString(dest));
            assertTrue(!Files.exists(dest.resolveSibling("m.gguf.part.etag")), "cleaned up");
        }
    }

    @Test
    void aWrongSha256FailsAndLeavesNothingBehind(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            String wrong = "0".repeat(64);

            var failure =
                    assertThrows(
                            IOException.class,
                            () ->
                                    Fetch.download(
                                            server.url("/m.gguf"),
                                            dest,
                                            PAYLOAD.length(),
                                            wrong,
                                            Map.of()));
            assertTrue(failure.getMessage().contains("sha256 mismatch"), failure.getMessage());
            assertTrue(Files.notExists(dest), "dest is either complete or absent");
            // the law is stronger than that: the partial is DELETED, because resuming bytes that
            // can never match the hash would fail every future attempt the same way
            assertTrue(Files.notExists(dest.resolveSibling("m.gguf.part")));
        }
    }

    @Test
    void aResumeFetchesOnlyWhatIsMissing(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");
            int kept = PAYLOAD.length() / 2;
            Files.writeString(dest.resolveSibling("m.gguf.part"), PAYLOAD.substring(0, kept));

            Fetch.download(
                    server.url("/m.gguf"), dest, PAYLOAD.length(), sha256(PAYLOAD), Map.of());

            assertEquals(PAYLOAD, Files.readString(dest));
            assertEquals(
                    "bytes=" + kept + "-",
                    server.lastRange("/m.gguf"),
                    "the fetch continued where the .part ended");
        }
    }

    @Test
    void anUndersizedResponseFailsRatherThanTruncating(@TempDir Path dir) throws IOException {
        try (FileServer server = FileServer.start().serve("/m.gguf", PAYLOAD)) {
            Path dest = dir.resolve("m.gguf");

            assertThrows(
                    IOException.class,
                    () ->
                            Fetch.download(
                                    server.url("/m.gguf"),
                                    dest,
                                    PAYLOAD.length() + 1000,
                                    null,
                                    Map.of()));
            assertTrue(Files.notExists(dest));
        }
    }
}
