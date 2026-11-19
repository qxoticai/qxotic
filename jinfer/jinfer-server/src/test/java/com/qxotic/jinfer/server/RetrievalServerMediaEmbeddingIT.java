package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Media embeddings over HTTP with EmbeddingGemma 2 and its projector: content-part arrays (text,
 * image, audio) each embedded as one vector, Matryoshka {@code dimensions}, OpenAI's base64 form,
 * and a modality the model does not take refused before its decoder runs.
 */
@Tag("integration")
@SuppressWarnings("unchecked")
final class RetrievalServerMediaEmbeddingIT {

    private static final String MODEL =
            "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    private static final String MMPROJ = "hf.co/unsloth/embeddinggemma-2-GGUF/mmproj-BF16.gguf";

    private final HttpClient client = HttpClient.newHttpClient();

    private static LoadedEmbedder<?> load(Arena arena) throws Exception {
        return Models.loadEmbedder(
                TestModels.require(MODEL),
                arena,
                Map.of("media", TestModels.require(MMPROJ)),
                null);
    }

    private record Reply(int status, Map<String, Object> body) {
        String error() {
            Object error = body.get("error");
            return error instanceof Map<?, ?> map ? (String) map.get("message") : null;
        }

        List<Map<String, Object>> data() {
            return (List<Map<String, Object>>) body.get("data");
        }
    }

    private Reply post(String url, String json) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(url))
                                .POST(HttpRequest.BodyPublishers.ofString(json))
                                .header("Content-Type", "application/json")
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        return new Reply(
                response.statusCode(),
                (Map<String, Object>) com.qxotic.jinfer.chat.JsonCodec.parse(response.body()));
    }

    @Test
    void partsArraysEmbedOneVectorEach() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            LoadedEmbedder<?> embedder = load(arena);
            try (var server = RetrievalServer.start(embedder, 2048, MODEL, ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort();
                String cat =
                        Base64.getEncoder()
                                .encodeToString(
                                        Files.readAllBytes(
                                                TestModels.fixture("lfm3/images/" + "cat.png")));
                String wav =
                        Base64.getEncoder()
                                .encodeToString(
                                        Files.readAllBytes(
                                                Path.of(
                                                        "../../test-fixtures/parakeet/"
                                                                + "ls0.wav")));

                // a text and an image in one array: one vector
                Reply interleaved =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [[{\"type\": \"text\", \"text\": \"a photo\"},"
                                        + " {\"type\": \"image_url\", \"image_url\": {\"url\":"
                                        + " \"data:image/png;base64,"
                                        + cat
                                        + "\"}}]], \"dimensions\": 256}");
                assertEquals(200, interleaved.status(), String.valueOf(interleaved.body()));
                assertEquals(1, interleaved.data().size());
                assertEquals(256, floats(interleaved.data().getFirst()).size());

                // an image alone
                Reply image =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [[{\"type\": \"image_url\", \"image_url\": {\"url\":"
                                        + " \"data:image/png;base64,"
                                        + cat
                                        + "\"}}]], \"dimensions\": 256}");
                assertEquals(200, image.status(), String.valueOf(image.body()));
                assertEquals(256, floats(image.data().getFirst()).size());

                // a clip alone
                Reply audio =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [[{\"type\": \"input_audio\", \"input_audio\":"
                                        + " {\"data\": \""
                                        + wav
                                        + "\", \"format\": \"wav\"}}]], \"dimensions\": 256}");
                assertEquals(200, audio.status(), String.valueOf(audio.body()));
                assertEquals(256, floats(audio.data().getFirst()).size());

                // an array per input, and a bare string beside: two vectors, in order
                Reply batch =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [[{\"type\": \"text\", \"text\": \"one\"}],"
                                        + " \"two\"]}");
                assertEquals(200, batch.status(), String.valueOf(batch.body()));
                assertEquals(2, batch.data().size());
                // JSON numbers parse as longs: compare the values, not the box types
                assertEquals(
                        List.of(0, 1),
                        batch.data().stream()
                                .map(d -> ((Number) d.get("index")).intValue())
                                .toList());
                assertEquals(768, floats(batch.data().getFirst()).size(), "the model's own width");

                // OpenAI's base64 form: little-endian float32
                Reply base64 =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [\"hi\"], \"encoding_format\": \"base64\","
                                        + " \"dimensions\": 128}");
                assertEquals(200, base64.status(), String.valueOf(base64.body()));
                String encoded = (String) base64.data().getFirst().get("embedding");
                assertEquals(
                        128,
                        Base64.getDecoder().decode(encoded).length / Float.BYTES,
                        "128 float32 values, little-endian");
            }
        } finally {
            Arenas.close(arena);
        }
    }

    /** A modality the model cannot read is refused naming what it does, before any decoder runs. */
    @Test
    void anUnreadModalityIsRefusedNamingTheProjector() throws Exception {
        Arena arena = Arenas.newCrossThread();
        try {
            // the same checkpoint WITHOUT its projector: text only
            LoadedEmbedder<?> bare = Models.loadEmbedder(TestModels.require(MODEL), arena);
            try (var server = RetrievalServer.start(bare, 2048, MODEL, ServerConfig.local(0))) {
                String url = "http://127.0.0.1:" + server.address().getPort();
                Reply refused =
                        post(
                                url + "/v1/embeddings",
                                "{\"input\": [[{\"type\": \"image_url\", \"image_url\": {\"url\":"
                                        + " \"data:image/png;base64,AAAA\"}}]]}");
                assertEquals(400, refused.status(), String.valueOf(refused.body()));
                assertTrue(
                        refused.error().contains("this model embeds text, not image"),
                        refused.error());
                assertTrue(refused.error().contains("--with media=<mmproj>"), refused.error());
            }
        } finally {
            Arenas.close(arena);
        }
    }

    private static List<Float> floats(Map<String, Object> datum) {
        return (List<Float>) datum.get("embedding");
    }
}
