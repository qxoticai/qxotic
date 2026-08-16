package com.qxotic.jinfer.langchain4j;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.segment.TextSegment;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Public-API reranker throughput bench, optionally compared with a {@code /v1/rerank} server. */
public final class ScoringBench {

    private static final int[] DOCUMENT_COUNTS = {4, 16};
    private static final int REPS = 3;
    private static final String QUERY = "When was the Eiffel Tower built and how tall is it?";
    private static final String REF =
            "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf";

    @Test
    @Tag("bench")
    void run() throws Exception {
        String url = System.getProperty("jinfer.args", "").trim();
        try (JinferScoringModel model =
                JinferScoringModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(2048)
                        .build()) {
            for (int count : DOCUMENT_COUNTS) {
                List<TextSegment> documents = corpus(count);
                Runnable leg = () -> model.scoreAll(documents, QUERY);
                leg.run();
                double[] times = new double[REPS];
                for (int i = 0; i < REPS; i++) times[i] = timeMs(leg);
                double milliseconds = median(times);
                System.out.printf(
                        "k=%-3d jinfer %8.1f ms (%5.1f pairs/s)%n",
                        count, milliseconds, 1000.0 * count / milliseconds);

                if (!url.isEmpty()) {
                    List<Double> local = model.scoreAll(documents, QUERY).content();
                    HttpResult remote = rerank(url, QUERY, documents);
                    System.out.printf(
                            "k=%-3d http   %8.1f ms (%5.1f pairs/s) rank-agreement=%.3f%n",
                            count,
                            remote.milliseconds(),
                            1000.0 * count / remote.milliseconds(),
                            spearman(local, remote.scores()));
                }
            }
        }
    }

    private static List<TextSegment> corpus(int count) {
        String[] seeds = {
            "The Eiffel Tower is a wrought-iron lattice tower in Paris, completed in 1889 and"
                    + " standing roughly 330 metres tall including antennas.",
            "Photosynthesis converts light energy into chemical energy inside chloroplasts.",
            "The recipe calls for flour, salt, eggs and melted butter.",
            "Interest rates influence bond prices inversely."
        };
        List<TextSegment> documents = new ArrayList<>(count);
        for (int i = 0; i < count; i++)
            documents.add(TextSegment.from(seeds[i % seeds.length] + " (variant " + i + ")"));
        return documents;
    }

    private static HttpResult rerank(String base, String query, List<TextSegment> documents)
            throws Exception {
        String body =
                Json.stringify(
                        Map.of(
                                "query",
                                query,
                                "documents",
                                documents.stream().map(TextSegment::text).toList()));
        HttpRequest request =
                HttpRequest.newBuilder(URI.create(base + "/v1/rerank"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body))
                        .build();
        HttpClient client = HttpClient.newHttpClient();
        client.send(request, HttpResponse.BodyHandlers.discarding());
        long start = System.nanoTime();
        HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() / 100 != 2)
            throw new IllegalStateException(response.statusCode() + ": " + response.body());
        @SuppressWarnings("unchecked")
        Map<String, Object> root = (Map<String, Object>) Json.parse(response.body());
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> data = (List<Map<String, Object>>) root.get("data");
        List<Double> scores =
                data.stream()
                        .sorted(
                                Comparator.comparingInt(
                                        row -> ((Number) row.get("index")).intValue()))
                        .map(row -> ((Number) row.get("relevance_score")).doubleValue())
                        .toList();
        return new HttpResult((System.nanoTime() - start) / 1e6, scores);
    }

    private static double timeMs(Runnable action) {
        long start = System.nanoTime();
        action.run();
        return (System.nanoTime() - start) / 1e6;
    }

    private static double median(double[] values) {
        double[] sorted = values.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    private static double spearman(List<Double> left, List<Double> right) {
        if (left.size() != right.size() || left.size() < 2) return Double.NaN;
        int[] a = ranks(left), b = ranks(right);
        long squaredDifference = 0;
        for (int i = 0; i < a.length; i++) {
            long difference = a[i] - b[i];
            squaredDifference += difference * difference;
        }
        return 1.0 - 6.0 * squaredDifference / (a.length * (a.length * a.length - 1.0));
    }

    private static int[] ranks(List<Double> values) {
        Integer[] order = new Integer[values.size()];
        for (int i = 0; i < order.length; i++) order[i] = i;
        Arrays.sort(order, Comparator.comparingDouble(values::get));
        int[] ranks = new int[order.length];
        for (int i = 0; i < order.length; i++) ranks[order[i]] = i;
        return ranks;
    }

    private record HttpResult(double milliseconds, List<Double> scores) {}
}
