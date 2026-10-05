package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Base64;
import java.util.List;
import java.util.Map;
import java.util.function.Consumer;
import java.util.function.DoubleConsumer;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

/** The retrieval transport over fake models: the wire shapes, the refusals, the probes. */
class RetrievalServerTest {

    private static final String MODEL = "fake-retriever.gguf";
    private final HttpClient client = HttpClient.newHttpClient();
    private AutoCloseable running;
    private String base;

    /** One vector per text: {@code [length, 0, 1, 2, ...]}, as wide as asked. */
    static final class FakeEmbedder implements RetrievalServer.Embedder {
        final int minimumDimension;
        final List<List<String>> calls = new ArrayList<>();
        boolean closed;

        FakeEmbedder(int minimumDimension) {
            this.minimumDimension = minimumDimension;
        }

        public int dimension() {
            return 8;
        }

        public int minimumDimension() {
            return minimumDimension;
        }

        public String queryPrefix() {
            return "query: ";
        }

        public String documentPrefix() {
            return "document: ";
        }

        public int contextCapacity() {
            return 16;
        }

        public int tokens(String text) {
            return text.split(" ").length + 1; // one framing token, as Qwen3's trailing EOS
        }

        public int embed(List<String> texts, int dimension, Consumer<float[]> sink) {
            calls.add(List.copyOf(texts));
            int total = 0;
            for (String text : texts) {
                float[] vector = new float[dimension];
                vector[0] = text.length();
                for (int i = 1; i < dimension; i++) vector[i] = i - 1 + 0.5f;
                sink.accept(vector);
                total += tokens(text);
            }
            return total;
        }

        public void close() {
            closed = true;
        }
    }

    /** Scores a document by how many query words it holds; "LONG" overflows the context. */
    static final class FakeRanker implements RetrievalServer.Ranker {
        boolean closed;

        public int contextCapacity() {
            return 16;
        }

        public int score(String query, List<String> documents, DoubleConsumer sink) {
            int tokens = query.split(" ").length;
            for (int i = 0; i < documents.size(); i++) {
                String document = documents.get(i);
                if (document.contains("LONG"))
                    throw new IllegalArgumentException(
                            "document "
                                    + i
                                    + " frames to 99 tokens, over the 16-token context - raise"
                                    + " contextCapacity(...) or chunk smaller");
                int hits = 0;
                for (String word : query.split(" ")) if (document.contains(word)) hits++;
                sink.accept(hits);
                tokens += document.split(" ").length;
            }
            return tokens;
        }

        public void close() {
            closed = true;
        }
    }

    @AfterEach
    void stop() throws Exception {
        if (running != null) running.close();
    }

    private void serve(RetrievalServer.Embedder embedder) throws Exception {
        var server = RetrievalServer.start(embedder, MODEL, ServerConfig.local(0));
        running = server;
        base = "http://127.0.0.1:" + server.address().getPort();
    }

    private void serve(RetrievalServer.Ranker ranker) throws Exception {
        var server = RetrievalServer.start(ranker, MODEL, ServerConfig.local(0));
        running = server;
        base = "http://127.0.0.1:" + server.address().getPort();
    }

    private HttpResponse<String> post(String path, String body) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(base + path))
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build(),
                HttpResponse.BodyHandlers.ofString());
    }

    private HttpResponse<String> get(String path) throws Exception {
        return client.send(
                HttpRequest.newBuilder(URI.create(base + path)).GET().build(),
                HttpResponse.BodyHandlers.ofString());
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> json(HttpResponse<String> response) {
        return (Map<String, Object>) JsonCodec.parse(response.body());
    }

    /** A 400 naming {@code param}, with {@code fragment} in its message. */
    private void refused(String path, String body, String param, String fragment) throws Exception {
        HttpResponse<String> response = post(path, body);
        assertEquals(400, response.statusCode(), body + " -> " + response.body());
        @SuppressWarnings("unchecked")
        Map<String, Object> error = (Map<String, Object>) json(response).get("error");
        assertEquals(param, error.get("param"), response.body());
        assertTrue(((String) error.get("message")).contains(fragment), response.body());
    }

    @Test
    @SuppressWarnings("unchecked")
    void embeddingsAnswerInOpenAiShapeOneBatchPerRequest() throws Exception {
        FakeEmbedder embedder = new FakeEmbedder(4);
        serve(embedder);
        HttpResponse<String> response =
                post("/v1/embeddings", "{\"input\":[\"a b\",\"c\"],\"model\":\"" + MODEL + "\"}");
        assertEquals(200, response.statusCode(), response.body());
        Map<String, Object> body = json(response);
        assertEquals("list", body.get("object"));
        assertEquals(MODEL, body.get("model"));
        assertEquals(Map.of("prompt_tokens", 5L, "total_tokens", 5L), body.get("usage"));
        List<Object> data = (List<Object>) body.get("data");
        assertEquals(2, data.size());
        Map<String, Object> second = (Map<String, Object>) data.get(1);
        assertEquals("embedding", second.get("object"));
        assertEquals(1L, second.get("index"));
        List<Object> vector = (List<Object>) second.get("embedding");
        assertEquals(8, vector.size());
        assertEquals(1.0, ((Number) vector.get(0)).doubleValue());
        assertEquals(1.5, ((Number) vector.get(2)).doubleValue());
        assertEquals(List.of(List.of("a b", "c")), embedder.calls, "one batch for the request");
        assertTrue(response.body().startsWith("{\"object\":\"list\",\"data\":"), response.body());
    }

    @Test
    @SuppressWarnings("unchecked")
    void base64IsLittleEndianFloat32AsTheOpenAiClientsDecodeIt() throws Exception {
        serve(new FakeEmbedder(4));
        Map<String, Object> body =
                json(post("/v1/embeddings", "{\"input\":\"abc\",\"encoding_format\":\"base64\"}"));
        String encoded =
                (String)
                        ((Map<String, Object>) ((List<Object>) body.get("data")).get(0))
                                .get("embedding");
        ByteBuffer bytes = ByteBuffer.wrap(Base64.getDecoder().decode(encoded));
        bytes.order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(8 * Float.BYTES, bytes.remaining());
        assertEquals(3f, bytes.getFloat(0));
        assertEquals(0.5f, bytes.getFloat(Float.BYTES));
    }

    @Test
    @SuppressWarnings("unchecked")
    void dimensionsShortenOnlyAMatryoshkaModelWithinItsRange() throws Exception {
        FakeEmbedder embedder = new FakeEmbedder(4);
        serve(embedder);
        Map<String, Object> body =
                json(post("/v1/embeddings", "{\"input\":\"a\",\"dimensions\":4}"));
        Map<String, Object> first = (Map<String, Object>) ((List<Object>) body.get("data")).get(0);
        assertEquals(4, ((List<Object>) first.get("embedding")).size());
        refused("/v1/embeddings", "{\"input\":\"a\",\"dimensions\":2}", "dimensions", "[4, 8]");
        refused("/v1/embeddings", "{\"input\":\"a\",\"dimensions\":9}", "dimensions", "[4, 8]");
        refused(
                "/v1/embeddings",
                "{\"input\":\"a\",\"dimensions\":\"x\"}",
                "dimensions",
                "integer");
    }

    @Test
    void aFixedWidthModelRefusesDimensionsItCannotHonor() throws Exception {
        serve(new FakeEmbedder(8));
        refused(
                "/v1/embeddings",
                "{\"input\":\"a\",\"dimensions\":4}",
                "dimensions",
                "fixed 8 dimensions");
        assertEquals(
                200, post("/v1/embeddings", "{\"input\":\"a\",\"dimensions\":8}").statusCode());
        // an explicit null is "unset", as OpenAI's SDKs send an omitted argument
        assertEquals(
                200, post("/v1/embeddings", "{\"input\":\"a\",\"dimensions\":null}").statusCode());
    }

    @Test
    void malformedInputIsRefusedNamingTheField() throws Exception {
        FakeEmbedder embedder = new FakeEmbedder(4);
        serve(embedder);
        refused("/v1/embeddings", "{}", "input", "input is required");
        refused("/v1/embeddings", "{\"input\":\"\"}", "input", "must not be empty");
        refused("/v1/embeddings", "{\"input\":[]}", "input", "empty array");
        refused("/v1/embeddings", "{\"input\":[\"a\",\"\"]}", "input", "input[1] must not be");
        refused("/v1/embeddings", "{\"input\":[\"a\",3]}", "input", "token-id input");
        refused("/v1/embeddings", "{\"input\":[[1,2]]}", "input", "token-id input");
        refused("/v1/embeddings", "{\"input\":[\"a\",{}]}", "input", "input[1] must be a string");
        refused("/v1/embeddings", "{\"input\":{}}", "input", "a string or an array");
        refused(
                "/v1/embeddings",
                "{\"input\":[\"a\",\"" + "w ".repeat(20) + "\"]}",
                "input",
                "input[1] is 21 tokens, over this server's 16-token context");
        refused(
                "/v1/embeddings",
                "{\"input\":\"a\",\"encoding_format\":\"int8\"}",
                "encoding_format",
                "float or base64");
        refused(
                "/v1/embeddings",
                "{\"input\":\"a\",\"input_type\":\"passage\"}",
                "input_type",
                "query or document");
        refused("/v1/embeddings", "{\"input\":\"a\",\"model\":7}", "model", "must be a string");
        assertEquals(400, post("/v1/embeddings", "not json").statusCode());
        assertTrue(embedder.calls.isEmpty(), "a refused request computes nothing");

        HttpResponse<String> other = post("/v1/embeddings", "{\"input\":\"a\",\"model\":\"gpt\"}");
        assertEquals(404, other.statusCode());
        assertTrue(other.body().contains("this server serves " + MODEL), other.body());
        assertEquals(200, post("/v1/embeddings", "{\"input\":\"a\",\"model\":\"\"}").statusCode());
    }

    @Test
    void inputTypeAppliesTheModelCardsRetrievalPrefix() throws Exception {
        FakeEmbedder embedder = new FakeEmbedder(4);
        serve(embedder);
        post("/v1/embeddings", "{\"input\":\"cats\",\"input_type\":\"query\"}");
        post("/v1/embeddings", "{\"input\":[\"cats\"],\"input_type\":\"document\"}");
        post("/v1/embeddings", "{\"input\":\"cats\"}");
        assertEquals(
                List.of(List.of("query: cats"), List.of("document: cats"), List.of("cats")),
                embedder.calls);
    }

    @Test
    @SuppressWarnings("unchecked")
    void rerankOrdersByRelevanceCutsToTopNAndReturnsDocumentsOnRequest() throws Exception {
        serve(new FakeRanker());
        String request =
                "{\"query\":\"red apple\",\"documents\":[\"blue sky\",\"red apple pie\","
                        + "{\"text\":\"a red car\"},\"green apple\"],\"top_n\":3,"
                        + "\"return_documents\":true}";
        for (String path : List.of("/v1/rerank", "/rerank", "/v1/reranking", "/reranking")) {
            HttpResponse<String> response = post(path, request);
            assertEquals(200, response.statusCode(), path + ": " + response.body());
            Map<String, Object> body = json(response);
            assertEquals(MODEL, body.get("model"));
            List<Object> results = (List<Object>) body.get("results");
            assertEquals(3, results.size());
            Map<String, Object> best = (Map<String, Object>) results.get(0);
            assertEquals(1L, best.get("index"));
            assertEquals(2.0, ((Number) best.get("relevance_score")).doubleValue());
            assertEquals(Map.of("text", "red apple pie"), best.get("document"));
            // a tie keeps input order: "a red car" (2) before "green apple" (3)
            assertEquals(2L, ((Map<String, Object>) results.get(1)).get("index"));
            assertEquals(3L, ((Map<String, Object>) results.get(2)).get("index"));
            assertEquals(Map.of("prompt_tokens", 12L, "total_tokens", 12L), body.get("usage"));
        }
        Map<String, Object> bare =
                json(post("/v1/rerank", "{\"query\":\"red\",\"documents\":[\"red\",\"blue\"]}"));
        List<Object> results = (List<Object>) bare.get("results");
        assertEquals(2, results.size(), "top_n defaults to every document");
        assertFalse(((Map<String, Object>) results.get(0)).containsKey("document"));
        Map<String, Object> none = json(post("/v1/rerank", "{\"query\":\"q\",\"documents\":[]}"));
        assertEquals(List.of(), none.get("results"));
    }

    @Test
    void malformedRerankRequestsAreRefusedNamingTheField() throws Exception {
        serve(new FakeRanker());
        refused("/v1/rerank", "{\"documents\":[\"a\"]}", "query", "non-empty string");
        refused("/v1/rerank", "{\"query\":\" \",\"documents\":[\"a\"]}", "query", "non-empty");
        refused("/v1/rerank", "{\"query\":\"q\"}", "documents", "documents is required");
        refused("/v1/rerank", "{\"query\":\"q\",\"documents\":\"a\"}", "documents", "an array");
        refused("/v1/rerank", "{\"query\":\"q\",\"documents\":[1]}", "documents", "documents[0]");
        refused(
                "/v1/rerank",
                "{\"query\":\"q\",\"documents\":[\"a\"],\"top_n\":0}",
                "top_n",
                "at least 1");
        refused(
                "/v1/rerank",
                "{\"query\":\"q\",\"documents\":[\"a\"],\"return_documents\":\"yes\"}",
                "return_documents",
                "a boolean");
        // the recipe's overflow names the document; its Java remedy is not the client's
        HttpResponse<String> tooLong =
                post("/v1/rerank", "{\"query\":\"q\",\"documents\":[\"a\",\"LONG\"]}");
        assertEquals(400, tooLong.statusCode());
        assertTrue(tooLong.body().contains("document 1 frames to 99 tokens"), tooLong.body());
        assertTrue(tooLong.body().contains("\"param\":\"documents\""), tooLong.body());
        assertFalse(tooLong.body().contains("contextCapacity"), tooLong.body());
    }

    @Test
    @SuppressWarnings("unchecked")
    void probesAndUnknownPathsAnswerAsOnTheOtherServers() throws Exception {
        FakeEmbedder embedder = new FakeEmbedder(4);
        serve(embedder);
        assertEquals("{\"status\":\"ok\",\"busy\":false,\"queued\":0}", get("/health").body());
        Map<String, Object> props = json(get("/props"));
        assertEquals("embedding", props.get("task"));
        assertEquals(16L, props.get("n_ctx"));
        assertEquals(8L, props.get("dimension"));
        assertEquals(4L, props.get("min_dimension"));
        Map<String, Object> models = json(get("/v1/models"));
        Map<String, Object> card = (Map<String, Object>) ((List<Object>) models.get("data")).get(0);
        assertEquals(MODEL, card.get("id"));
        assertEquals(200, get("/v1/models/" + MODEL).statusCode());
        assertEquals(404, get("/v1/models/other").statusCode());

        post("/v1/embeddings", "{\"input\":[\"a\",\"b\"]}");
        String metrics = get("/metrics").body();
        assertTrue(metrics.contains("jinfer_embedding_requests_completed_total 1"), metrics);
        assertTrue(metrics.contains("jinfer_embedding_inputs_total 2"), metrics);
        assertTrue(metrics.contains("jinfer_prompt_tokens_total 4"), metrics);

        HttpResponse<String> chat = post("/v1/chat/completions", "{}");
        assertEquals(404, chat.statusCode());
        assertTrue(chat.body().contains("this server serves POST /v1/embeddings"), chat.body());
        assertTrue(chat.headers().firstValue("Content-Type").orElse("").contains("json"));
        assertEquals(404, post("/v1/embeddingsX", "{}").statusCode());
        assertEquals(405, get("/v1/embeddings").statusCode());

        ((AutoCloseable) running).close();
        running = null;
        assertTrue(embedder.closed, "the server's state closes with it");
    }
}
