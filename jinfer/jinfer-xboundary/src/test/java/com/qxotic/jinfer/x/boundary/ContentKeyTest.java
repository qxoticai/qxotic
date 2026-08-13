package com.qxotic.jinfer.x.boundary;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.time.Duration;
import org.junit.jupiter.api.Test;

class ContentKeyTest {

    @Test
    void sha256MatchesTheReferenceVector() {
        assertEquals(
                "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
                ContentKey.sha256("abc".getBytes(StandardCharsets.UTF_8)).toString());
    }

    @Test
    void digestBytesRoundTripsTheSha256Digest() throws Exception {
        byte[] source = {1, 2, 3};
        assertArrayEquals(
                MessageDigest.getInstance("SHA-256").digest(source),
                ContentKey.sha256(source).digestBytes());
        assertThrows(
                IllegalStateException.class, () -> new ContentKey("my-model-v1").digestBytes());
    }

    @Test
    void keySemantics() {
        assertThrows(IllegalArgumentException.class, () -> new ContentKey(""));
        var video = ContentKey.sha256(new byte[] {1, 2, 3});
        var a = new FrameRef(video, Duration.ofNanos(1_500_000_000L));
        assertEquals(a, new FrameRef(video, Duration.ofMillis(1500)));
        assertNotEquals(a, new FrameRef(video, Duration.ofNanos(1_500_000_001L)));
        assertEquals(video + "#t=1500000000", a.toString());
    }
}
