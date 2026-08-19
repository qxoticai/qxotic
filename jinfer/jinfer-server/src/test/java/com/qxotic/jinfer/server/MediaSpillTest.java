package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import java.nio.file.Files;
import java.security.MessageDigest;
import java.util.Base64;
import java.util.HexFormat;
import java.util.Random;
import org.junit.jupiter.api.Test;

class MediaSpillTest {

    private static String dataUri(byte[] payload) {
        return "data:video/mp4;base64," + Base64.getEncoder().encodeToString(payload);
    }

    private static byte[] randomBytes(int size, long seed) {
        byte[] bytes = new byte[size];
        new Random(seed).nextBytes(bytes);
        return bytes;
    }

    @Test
    void spillsBytesAndHashesInOnePass() throws Exception {
        byte[] payload = randomBytes(300_000, 42); // spans several decode chunks
        MediaSpill.Spilled spilled = MediaSpill.base64Video(dataUri(payload), "video_url");
        try {
            assertArrayEquals(payload, Files.readAllBytes(spilled.file()));
            String expectedSha =
                    "sha256:"
                            + HexFormat.of()
                                    .formatHex(
                                            MessageDigest.getInstance("SHA-256").digest(payload));
            assertEquals(new ContentKey(expectedSha), spilled.key());
        } finally {
            MediaSpill.deleteQuietly(spilled.file());
        }
        assertTrue(Files.notExists(spilled.file()));
    }

    @Test
    void emptyPayloadIsAccepted() throws Exception {
        MediaSpill.Spilled spilled = MediaSpill.base64Video("data:video/mp4;base64,", "video_url");
        try {
            assertEquals(0, Files.size(spilled.file()));
        } finally {
            MediaSpill.deleteQuietly(spilled.file());
        }
    }

    @Test
    void rejectsNonDataUri() {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MediaSpill.base64Video("https://example.com/clip.mp4", "video_url"));
        assertTrue(failure.getMessage().contains("must be a data: URI"));
    }

    @Test
    void rejectsNonBase64DataUri() {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MediaSpill.base64Video("data:video/mp4,Ubah", "video_url"));
        assertTrue(failure.getMessage().contains("must be base64-encoded"));
    }

    @Test
    void rejectsMalformedBase64() {
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                MediaSpill.base64Video(
                                        "data:video/mp4;base64,!!@@##$$", "video_url"));
        assertTrue(failure.getMessage().contains("malformed"));
    }

    @Test
    void rejectsPayloadOverTheLimit() {
        // Just over the cap; the encoded-length pre-check rejects before any file is created.
        byte[] payload = randomBytes((int) MediaSpill.MAX_INLINE_VIDEO_BYTES + 1, 7);
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MediaSpill.base64Video(dataUri(payload), "video_url"));
        assertTrue(failure.getMessage().contains("inline video limit"));
    }

    @Test
    void acceptsPayloadAtTheLimit() throws Exception {
        byte[] payload = randomBytes((int) MediaSpill.MAX_INLINE_VIDEO_BYTES, 7);
        MediaSpill.Spilled spilled = MediaSpill.base64Video(dataUri(payload), "video_url");
        try {
            assertEquals(MediaSpill.MAX_INLINE_VIDEO_BYTES, Files.size(spilled.file()));
        } finally {
            MediaSpill.deleteQuietly(spilled.file());
        }
    }

    @Test
    void overLimitErrorMentionsTheField() {
        byte[] payload = randomBytes((int) MediaSpill.MAX_INLINE_VIDEO_BYTES + 1, 7);
        IllegalArgumentException failure =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> MediaSpill.base64Video(dataUri(payload), "video_url"));
        assertTrue(failure.getMessage().startsWith("video_url exceeds"));
    }
}
