package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.ModelProvider;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** The retrieval transport over the real checkpoints it is started with by {@code server}. */
@Tag("integration")
class RetrievalServerIntegrationTest {

    private static final String EMBEDDING =
            "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";
    private static final String RERANKER =
            "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf";
    private static final String COLBERT =
            "hf.co/LiquidAI/LFM2.5-ColBERT-350M-GGUF/LFM2.5-ColBERT-350M-Q8_0.gguf";
    private static final String LFM_EMBEDDING =
            "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf";

    private static final String DOCUMENTS =
            "[\"Paris is the capital and largest city of France.\","
                    + "\"Bananas are rich in potassium.\","
                    + "\"Berlin is the capital of Germany.\"]";

    private final HttpClient client = HttpClient.newHttpClient();

    /** The header alone tells every retrieval checkpoint's face - what the CLI dispatches on. */
    @Test
    void theHeaderNamesEachCheckpointsFace() throws Exception {
        assertEquals(
                Optional.of(ModelProvider.Retrieval.EMBEDDING),
                Models.retrieval(TestModels.require(EMBEDDING)));
        assertEquals(
                Optional.of(ModelProvider.Retrieval.RERANKING),
                Models.retrieval(TestModels.require(RERANKER)));
        assertEquals(
                Optional.of(ModelProvider.Retrieval.RERANKING),
                Models.retrieval(TestModels.require(COLBERT)));
        assertEquals(
                Optional.of(ModelProvider.Retrieval.EMBEDDING),
                Models.retrieval(TestModels.require(LFM_EMBEDDING)));
    }

    @Test
    @SuppressWarnings("unchecked")
    void embeddingsRankTheRelatedSentenceFirst() throws Exception {
        Path path = TestModels.require(EMBEDDING);
        Arena arena = Arenas.newCrossThread();
        try {
            LoadedEmbedder<?> embedder = Models.loadEmbedder(path, arena);
            try (var server =
                    RetrievalServer.start(
                            embedder, 512, "qwen3-embedding", ServerConfig.local(0))) {
                String base = "http://127.0.0.1:" + server.address().getPort();
                Map<String, Object> body =
                        post(
                                base + "/v1/embeddings",
                                "{\"input\":"
                                        + DOCUMENTS
                                        + ",\"dimensions\":256,\"model\":\"qwen3-embedding\"}");
                List<Object> data = (List<Object>) body.get("data");
                assertEquals(3, data.size());
                double[][] vectors = new double[3][];
                for (int i = 0; i < 3; i++) {
                    List<Object> values =
                            (List<Object>) ((Map<String, Object>) data.get(i)).get("embedding");
                    assertEquals(256, values.size());
                    vectors[i] =
                            values.stream().mapToDouble(v -> ((Number) v).doubleValue()).toArray();
                    assertEquals(1.0, dot(vectors[i], vectors[i]), 1e-4, "unit length");
                }
                assertTrue(dot(vectors[0], vectors[2]) > dot(vectors[0], vectors[1]));
                Map<String, Object> usage = (Map<String, Object>) body.get("usage");
                assertTrue(((Number) usage.get("prompt_tokens")).intValue() > 20, usage.toString());
            }
        } finally {
            Arenas.close(arena);
        }
    }

    @Test
    void theQwen3JudgeScoresProbabilitiesAndColbertUnboundedSums() throws Exception {
        assertRanksParisFirst(RERANKER, 0, 1);
        // MaxSim sums one cosine per (padded, 32-row) query token: ranks, never thresholds
        assertRanksParisFirst(COLBERT, 1, 32);
    }

    @SuppressWarnings("unchecked")
    private void assertRanksParisFirst(String ref, double low, double high) throws Exception {
        Path path = TestModels.require(ref);
        Arena arena = Arenas.newCrossThread();
        try {
            LoadedReranker<?> reranker = Models.loadReranker(path, arena);
            try (var server =
                    RetrievalServer.start(reranker, 512, "reranker", ServerConfig.local(0))) {
                String base = "http://127.0.0.1:" + server.address().getPort();
                Map<String, Object> body =
                        post(
                                base + "/v1/rerank",
                                "{\"query\":\"What is the capital of France?\",\"documents\":"
                                        + DOCUMENTS
                                        + ",\"top_n\":2}");
                List<Object> results = (List<Object>) body.get("results");
                assertEquals(2, results.size());
                Map<String, Object> best = (Map<String, Object>) results.get(0);
                assertEquals(0L, best.get("index"), body.toString());
                double score = ((Number) best.get("relevance_score")).doubleValue();
                assertTrue(low < score && score <= high, ref + " scored " + score);
            }
        } finally {
            Arenas.close(arena);
        }
    }

    @SuppressWarnings("unchecked")
    private Map<String, Object> post(String uri, String body) throws Exception {
        HttpResponse<String> response =
                client.send(
                        HttpRequest.newBuilder(URI.create(uri))
                                .POST(HttpRequest.BodyPublishers.ofString(body))
                                .build(),
                        HttpResponse.BodyHandlers.ofString());
        assertEquals(200, response.statusCode(), response.body());
        return (Map<String, Object>) JsonCodec.parse(response.body());
    }

    private static double dot(double[] a, double[] b) {
        double sum = 0;
        for (int i = 0; i < a.length; i++) sum += a[i] * b[i];
        return sum;
    }
}
