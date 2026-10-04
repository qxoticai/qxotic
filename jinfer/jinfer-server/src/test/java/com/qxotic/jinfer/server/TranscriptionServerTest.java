package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.RuntimeState;
import com.qxotic.jinfer.Transcription;
import com.qxotic.jinfer.TranscriptionModel;
import com.qxotic.jinfer.codecs.AudioCodec;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jota.memory.MemoryArena;
import java.io.OutputStream;
import java.lang.foreign.MemorySegment;
import java.net.InetAddress;
import java.net.Socket;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
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

    static final class NoState extends RuntimeState {
        @Override
        protected void releaseResources() {}
    }

    /** A model that answers {@code transcribe} with whatever {@code reply} does. */
    record FakeModel(Function<float[], Transcription> reply)
            implements TranscriptionModel<Void, Void, NoState> {
        @Override
        public int sampleRate() {
            return 16000;
        }

        @Override
        public NoState newState() {
            return new NoState();
        }

        @Override
        public NoState newState(MemoryArena<MemorySegment> arena) {
            return new NoState();
        }

        @Override
        public Transcription transcribe(NoState state, float[] pcm) {
            return reply.apply(pcm);
        }

        @Override
        public Void configuration() {
            return null;
        }

        @Override
        public Void weights() {
            return null;
        }
    }

    record Answer(int status, String body) {}

    /**
     * Posts a short WAV over a raw socket; the body follows the headers after {@code pauseMillis},
     * like a slow link, so a body-read deadline that already expired is bound to fire.
     */
    private static Answer upload(TranscriptionServer.Running server, long pauseMillis)
            throws Exception {
        String boundary = "jinfer-test-boundary";
        byte[] wav = AudioCodec.wav(new Media.Audio(new float[1600], 16000, 1));
        byte[] body =
                concat(
                        ("--"
                                        + boundary
                                        + "\r\n"
                                        + "Content-Disposition: form-data; name=\"file\";"
                                        + " filename=\"a.wav\"\r\n"
                                        + "Content-Type: audio/wav\r\n\r\n")
                                .getBytes(StandardCharsets.UTF_8),
                        wav,
                        ("\r\n--" + boundary + "--\r\n").getBytes(StandardCharsets.UTF_8));
        try (Socket socket =
                new Socket(InetAddress.getLoopbackAddress(), server.address().getPort())) {
            socket.setSoTimeout(10_000);
            OutputStream out = socket.getOutputStream();
            out.write(
                    ("POST /v1/audio/transcriptions HTTP/1.1\r\n"
                                    + "Host: 127.0.0.1\r\n"
                                    + "Content-Type: multipart/form-data; boundary="
                                    + boundary
                                    + "\r\n"
                                    + "Content-Length: "
                                    + body.length
                                    + "\r\n"
                                    + "Connection: close\r\n\r\n")
                            .getBytes(StandardCharsets.US_ASCII));
            out.flush();
            Thread.sleep(pauseMillis);
            out.write(body);
            out.flush();
            String response =
                    new String(socket.getInputStream().readAllBytes(), StandardCharsets.UTF_8);
            assertTrue(response.startsWith("HTTP/1.1 "), "no response: '" + response + "'");
            return new Answer(
                    Integer.parseInt(response.substring(9, 12)),
                    response.substring(response.indexOf("\r\n\r\n") + 4));
        }
    }

    @Test
    void aDisabledRequestTimeoutDoesNotCutTheUploadShort() throws Exception {
        // "--request-timeout 0 disables": the body read is bounded by the write timeout instead
        ServerConfig config =
                ServerConfig.local(0)
                        .withLimits(ServerConfig.Limits.DEFAULTS.withRequestTimeout(Duration.ZERO));
        var model = new FakeModel(pcm -> new Transcription("hi", List.of()));
        try (var server = TranscriptionServer.start(model, "asr", config)) {
            Answer response = upload(server, 300);
            assertEquals(200, response.status(), response.body());
            assertTrue(response.body().contains("{\"text\":\"hi\"}"), response.body());
        }
    }

    @Test
    void aServerFaultIsA500ThatEchoesNothing() throws Exception {
        var model =
                new FakeModel(
                        pcm -> {
                            throw new IllegalStateException("internal detail");
                        });
        try (var server = TranscriptionServer.start(model, "asr", ServerConfig.local(0))) {
            Answer response = upload(server, 0);
            assertEquals(500, response.status(), response.body());
            assertTrue(response.body().contains("Internal server error"), response.body());
            assertFalse(response.body().contains("internal detail"), response.body());
        }
    }

    @Test
    void anUnknownModelIdIsNamedAsSuch() throws Exception {
        var model = new FakeModel(pcm -> Transcription.empty());
        try (var server = TranscriptionServer.start(model, "asr", ServerConfig.local(0))) {
            String base = "http://127.0.0.1:" + server.address().getPort();
            HttpClient client = HttpClient.newHttpClient();
            HttpResponse<String> unknown =
                    client.send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/models/other")).build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(404, unknown.statusCode());
            assertTrue(unknown.body().contains("Unknown model: other"), unknown.body());
            HttpResponse<String> wrongPath =
                    client.send(
                            HttpRequest.newBuilder(URI.create(base + "/v1/modelsXYZ")).build(),
                            HttpResponse.BodyHandlers.ofString());
            assertEquals(404, wrongPath.statusCode());
            assertTrue(wrongPath.body().contains("Not found"), wrongPath.body());
        }
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
