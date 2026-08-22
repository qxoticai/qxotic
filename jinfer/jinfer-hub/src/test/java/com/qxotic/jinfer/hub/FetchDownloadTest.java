package com.qxotic.jinfer.hub;

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
