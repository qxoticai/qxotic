package com.qxotic.jinfer.x.server;

import com.qxotic.jinfer.x.boundary.ContentKey;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;
import java.util.HexFormat;

/**
 * Spills an inline base64 media payload onto disk, streaming: the decoded bytes never sit on the
 * heap as a second full copy, the content hash is computed in the same pass, and the payload is
 * size-bounded. Inline video is the one media kind whose codec needs a seekable file (the ffmpeg
 * seam) - the wire owns that temp file.
 */
final class MediaSpill {

    /** Decoded-payload cap for inline base64 video. Beyond this, send a smaller clip. */
    static final long MAX_INLINE_VIDEO_BYTES = 20L * 1024 * 1024;

    /** Chunk of base64 text decoded per step; a multiple of 4 keeps every quantum intact. */
    private static final int CHUNK_CHARS = 4 * 8192;

    private MediaSpill() {}

    /** A spilled payload: the file it landed in and its content hash. The caller owns the file. */
    record Spilled(Path file, ContentKey key) {}

    /**
     * Decodes a {@code data:...;base64,...} URI into a fresh temp file, hashing while decoding.
     *
     * @throws IllegalArgumentException when the value is not a base64 data URI, the payload is
     *     malformed, or it exceeds {@link #MAX_INLINE_VIDEO_BYTES} once decoded
     * @throws IOException when the temp file cannot be written
     */
    static Spilled base64Video(String value, String field) throws IOException {
        Validation.require(
                value.startsWith("data:"),
                "%s must be a data: URI (the server does not fetch remote URLs)",
                field);
        int comma = value.indexOf(',');
        Validation.require(
                comma > 0 && value.substring(0, comma).endsWith(";base64"),
                "%s data: URI must be base64-encoded",
                field);
        String payload = value.substring(comma + 1);
        // Cheap upper bound on the decoded size, before touching disk: 3 decoded bytes per 4
        // encoded chars, plus 2 of slack for padding; the decode loop enforces the exact count.
        Validation.require(
                payload.length() / 4 * 3 <= MAX_INLINE_VIDEO_BYTES + 2,
                "%s exceeds the inline video limit of %d MB",
                field,
                MAX_INLINE_VIDEO_BYTES / 1024 / 1024);

        Path file = Files.createTempFile("jinfer-video", ".bin");
        try {
            MessageDigest digest = sha256();
            long total = 0;
            Base64.Decoder decoder = Base64.getDecoder();
            try (OutputStream out = Files.newOutputStream(file)) {
                for (int from = 0; from < payload.length(); from += CHUNK_CHARS) {
                    String chunk =
                            payload.substring(from, Math.min(from + CHUNK_CHARS, payload.length()));
                    byte[] decoded;
                    try {
                        decoded = decoder.decode(chunk);
                    } catch (IllegalArgumentException failure) {
                        throw new IllegalArgumentException(field + " base64 payload is malformed");
                    }
                    total += decoded.length;
                    if (total > MAX_INLINE_VIDEO_BYTES) {
                        throw new IllegalArgumentException(
                                "%s exceeds the inline video limit of %d MB"
                                        .formatted(field, MAX_INLINE_VIDEO_BYTES / 1024 / 1024));
                    }
                    digest.update(decoded);
                    out.write(decoded);
                }
            }
            return new Spilled(
                    file, new ContentKey("sha256:" + HexFormat.of().formatHex(digest.digest())));
        } catch (IOException | IllegalArgumentException failure) {
            deleteQuietly(file);
            throw failure;
        }
    }

    /** Best-effort delete with a {@code deleteOnExit} fallback, mirroring Generation's cleanup. */
    static void deleteQuietly(Path file) {
        try {
            Files.deleteIfExists(file);
        } catch (IOException ignored) {
            file.toFile().deleteOnExit();
        }
    }

    private static MessageDigest sha256() {
        try {
            return MessageDigest.getInstance("SHA-256");
        } catch (NoSuchAlgorithmException failure) {
            throw new AssertionError(failure);
        }
    }
}
