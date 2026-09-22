package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import com.qxotic.jinfer.Transcription;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TranscriptionServerTest {

    private static Duration ms(long millis) {
        return Duration.ofMillis(millis);
    }

    @Test
    void parsesTheMultipartShapeCurlAndOpenAiClientsSend() {
        String boundary = "------------------------d74496d66958873e";
        byte[] file = new byte[] {0, 1, 2, (byte) 0xFF, '\r', '\n', 3};
        byte[] body =
                concat(
                        ("--"
                                        + boundary
                                        + "\r\n"
                                        + "Content-Disposition: form-data; name=\"file\";"
                                        + " filename=\"jfk.wav\"\r\n"
                                        + "Content-Type: audio/wav\r\n\r\n")
                                .getBytes(StandardCharsets.UTF_8),
                        file,
                        ("\r\n--"
                                        + boundary
                                        + "\r\n"
                                        + "Content-Disposition: form-data;"
                                        + " name=\"response_format\"\r\n\r\n"
                                        + "verbose_json\r\n"
                                        + "--"
                                        + boundary
                                        + "--\r\n")
                                .getBytes(StandardCharsets.UTF_8));

        var parts = TranscriptionServer.Multipart.parse(body, boundary);
        assertArrayEquals(file, parts.get("file").content());
        assertEquals("verbose_json", parts.get("response_format").text());
    }

    @Test
    void boundaryComesFromTheContentTypeParameters() {
        assertEquals(
                "xyz", TranscriptionServer.Multipart.boundary("multipart/form-data; boundary=xyz"));
        assertEquals(
                "a b",
                TranscriptionServer.Multipart.boundary(
                        "multipart/form-data; charset=utf-8; boundary=\"a b\""));
        assertNull(TranscriptionServer.Multipart.boundary("application/json"));
        assertNull(TranscriptionServer.Multipart.boundary("multipart/form-data"));
    }

    @Test
    void tokensGroupIntoWordsAtLeadingSpaces() {
        Transcription transcription =
                new Transcription(
                        "And so, my",
                        List.of(
                                new Transcription.Token(" And", ms(200), ms(500), 0.9),
                                new Transcription.Token(" so", ms(500), ms(800), 0.8),
                                new Transcription.Token(",", ms(800), ms(900), 0.6),
                                new Transcription.Token(" my", ms(1000), ms(1200), 1.0)));
        List<Map<String, Object>> words = TranscriptionServer.words(transcription);
        assertEquals(3, words.size());
        assertEquals("And", words.get(0).get("word"));
        assertEquals("so,", words.get(1).get("word"));
        assertEquals(0.5, (double) words.get(1).get("start"));
        assertEquals(0.9, (double) words.get(1).get("end"));
        assertEquals(0.6, (double) words.get(1).get("confidence")); // min over the word's tokens
        assertEquals("my", words.get(2).get("word"));
    }

    private static byte[] concat(byte[]... chunks) {
        int length = 0;
        for (byte[] chunk : chunks) length += chunk.length;
        byte[] joined = new byte[length];
        int at = 0;
        for (byte[] chunk : chunks) {
            System.arraycopy(chunk, 0, joined, at, chunk.length);
            at += chunk.length;
        }
        return joined;
    }
}
