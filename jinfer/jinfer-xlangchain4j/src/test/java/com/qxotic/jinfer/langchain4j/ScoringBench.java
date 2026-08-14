// Reranker scoring bench: the shipped frame-reuse scorer vs a naive full-prefill baseline,
// and (optionally) any /v1/rerank server fed the IDENTICAL workload by this same harness.
//
//   jinfer legs:  mvn test -Dsurefire.excludedGroups= -Dgroups=bench -Dtest=ScoringBench \
//                     -pl jinfer-langchain4j
//   + llama.cpp:  add -Djinfer.args="http://127.0.0.1:8080"   (llama-server --reranking, same
//                 source weights + quant; note its GGUF is the rerank CONVERSION - cls_out head -
//                 while jinfer scores the stock causal GGUF; scores are compared for parity)
package com.qxotic.jinfer.langchain4j;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.boundary.Batch;
import com.qxotic.jinfer.x.boundary.Reranker;
import com.qxotic.jinfer.x.boundary.RuntimeState;
import com.qxotic.jinfer.x.chat.LoadedReranker;
import com.qxotic.jinfer.x.chat.Models;
import dev.langchain4j.data.segment.TextSegment;
import java.lang.foreign.Arena;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

public final class ScoringBench {

    static final int[] K = {4, 16};
    static final int REPS = 3;
    static final String[] QUERIES = {
        "When was the Eiffel Tower built and how tall is it?",
        "How do plants convert sunlight into chemical energy?",
    };

    private static final String REF =
            "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf";

    @Test
    @Tag("bench")
    void run() throws Exception {
        String url = System.getProperty("jinfer.args", "").trim();

        JinferScoringModel reuse =
                JinferScoringModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(2048)
                        .build();
        LoadedReranker<?> naive = Models.loadReranker(TestModels.require(REF), Arena.ofAuto());

        for (int k : K) {
            List<TextSegment> docs = corpus(k);
            String query = QUERIES[0];

            // cross-warm BOTH legs before timing EITHER, then interleave the timed reps -
            // leg order must not encode JIT tier state into the comparison
            Runnable reuseLeg = () -> reuse.scoreAll(docs, query);
            Runnable naiveLeg = () -> naiveScoreAll(naive, docs, query);
            for (int w = 0; w < 2; w++) {
                reuseLeg.run();
                naiveLeg.run();
            }
            double[] reuseTimes = new double[REPS];
            double[] naiveTimes = new double[REPS];
            for (int r = 0; r < REPS; r++) {
                reuseTimes[r] = timeMs(reuseLeg);
                naiveTimes[r] = timeMs(naiveLeg);
            }
            double reuseMs = median(reuseTimes);
            double naiveMs = median(naiveTimes);
            int frameTokens = frameTokens(naive, query);
            int docTokens = k * tailTokens(naive, docs.get(0));
            System.out.printf(
                    "k=%-3d reuse %8.1f ms (%5.1f pairs/s)   naive %8.1f ms (%5.1f pairs/s)"
                            + "   tokens: reuse=%d naive=%d%n",
                    k,
                    reuseMs,
                    1000.0 * k / reuseMs,
                    naiveMs,
                    1000.0 * k / naiveMs,
                    frameTokens + docTokens,
                    k * frameTokens + docTokens);

            if (!url.isEmpty()) {
                List<Double> mine = reuse.scoreAll(docs, query).content();
                double[] httpOut = httpLeg(url, query, docs);
                System.out.printf(
                        "k=%-3d %s %8.1f ms (%5.1f pairs/s)   rank-agreement(spearman)=%.3f%n",
                        k,
                        "http ",
                        httpOut[0],
                        1000.0 * k / httpOut[0],
                        spearman(mine, Arrays.stream(httpOut, 1, httpOut.length).boxed().toList()));
            }
        }
        reuse.close();
    }

    @Test
    @Tag("bench")
    void phases() throws Exception {
        phases(Models.loadReranker(TestModels.require(REF), Arena.ofAuto()));
    }

    static <S extends RuntimeState> void phases(LoadedReranker<S> loaded) {
        Reranker.CrossEncoder<S> reranker = (Reranker.CrossEncoder<S>) loaded.reranker();
        S state = newState(loaded);
        String query = QUERIES[0];
        TextSegment doc = corpus(1).get(0);
        // frame + tail ARE the full prompt: the seam after "<Document>:" is token-identical to
        // one continuous encoding, so "full" is simply both ingested from a reset state
        Batch frame = reranker.head(reranker.defaultInstruction(), query);
        Batch tail = reranker.document(doc.text());
        // warm
        for (int w = 0; w < 3; w++) {
            state.reset();
            ingest(loaded, state, frame);
            int p = state.position();
            state.resumeAt(p);
            ingest(loaded, state, tail);
            reranker.score(state);
            state.reset();
            ingest(loaded, state, frame);
            ingest(loaded, state, tail);
        }
        int n = 10;
        state.reset();
        ingest(loaded, state, frame);
        int p = state.position();
        long t0 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            state.resumeAt(p);
            ingest(loaded, state, tail);
        }
        double tailMs = (System.nanoTime() - t0) / 1e6 / n;
        t0 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            state.reset();
            ingest(loaded, state, tail);
        }
        double tailAt0Ms = (System.nanoTime() - t0) / 1e6 / n;
        t0 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            state.reset();
            ingest(loaded, state, frame);
            ingest(loaded, state, tail);
        }
        double fullMs = (System.nanoTime() - t0) / 1e6 / n;
        t0 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            state.reset();
            ingest(loaded, state, frame);
        }
        double frameMs = (System.nanoTime() - t0) / 1e6 / n;
        state.reset();
        ingest(loaded, state, frame);
        ingest(loaded, state, tail);
        t0 = System.nanoTime();
        for (int i = 0; i < n; i++) {
            reranker.score(state);
        }
        double scoreMs = (System.nanoTime() - t0) / 1e6 / n; // stop the clock BEFORE the sweep
        // depth sweep: does the tail cost scale with prefix depth (attention) or not (path)?
        for (int reps : new int[] {4, 16}) {
            state.reset();
            for (int i = 0; i < reps; i++) {
                ingest(loaded, state, frame);
            }
            int depth = state.position();
            long td = System.nanoTime();
            for (int i = 0; i < n; i++) {
                state.resumeAt(depth);
                ingest(loaded, state, tail);
            }
            System.out.printf("tail@%d %.1f ms%n", depth, (System.nanoTime() - td) / 1e6 / n);
        }
        System.out.printf(
                "phases: frame(%d tok) %.1f ms   tail(%d tok @pos %d) %.1f ms   tail@0 %.1f ms  "
                        + " full(%d tok) %.1f ms   score %.2f ms%n",
                frame.count(),
                frameMs,
                tail.count(),
                p,
                tailMs,
                tailAt0Ms,
                frame.count() + tail.count(),
                fullMs,
                scoreMs);
    }

    // ---- naive baseline: the pre-reuse scorer - full frame prefill per pair ----

    static <S extends RuntimeState> void naiveScoreAll(
            LoadedReranker<S> loaded, List<TextSegment> docs, String query) {
        Reranker.CrossEncoder<S> reranker = (Reranker.CrossEncoder<S>) loaded.reranker();
        String instruction = reranker.defaultInstruction();
        S state = newState(loaded);
        for (TextSegment doc : docs) {
            // token-identical to the reuse leg's frame + tail, but re-prefilled per pair: no
            // resumeAt, so every candidate pays the whole judge frame again
            state.reset();
            ingest(loaded, state, reranker.head(instruction, query));
            ingest(loaded, state, reranker.document(doc.text()));
            reranker.score(state);
        }
    }

    static <S extends RuntimeState> S newState(LoadedReranker<S> loaded) {
        return loaded.model().newState(Math.min(loaded.model().config().contextLength(), 4096));
    }

    static <S extends RuntimeState> void ingest(LoadedReranker<S> loaded, S state, Batch batch) {
        for (Batch chunk : Batch.prepare(List.of(batch), state.batchCapacity())) {
            loaded.model().ingest(state, chunk);
        }
    }

    // ---- shared workload ----

    static List<TextSegment> corpus(int k) {
        String[] seeds = {
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris,"
                    + " completed in 1889 as the entrance arch to the World's Fair and standing"
                    + " roughly 330 metres tall including antennas.",
            "Photosynthesis is the process by which green plants convert light energy into"
                    + " chemical energy, fixing carbon dioxide into sugars inside chloroplasts"
                    + " while releasing oxygen as a byproduct.",
            "The recipe calls for two cups of flour, a pinch of salt, three eggs and a slow"
                    + " fold of melted butter, rested for thirty minutes before baking at a"
                    + " moderate oven temperature.",
            "Interest rates influence bond prices inversely: when central banks raise rates,"
                    + " existing fixed-coupon bonds fall in value because newer issues offer"
                    + " higher yields to maturity.",
        };
        List<TextSegment> docs = new ArrayList<>(k);
        for (int i = 0; i < k; i++) {
            docs.add(TextSegment.from(seeds[i % seeds.length] + " (variant " + i + ")"));
        }
        return docs;
    }

    static int frameTokens(LoadedReranker<?> loaded, String query) {
        Reranker.CrossEncoder<?> reranker = (Reranker.CrossEncoder<?>) loaded.reranker();
        return reranker.head(reranker.defaultInstruction(), query).count();
    }

    static int tailTokens(LoadedReranker<?> loaded, TextSegment doc) {
        return ((Reranker.CrossEncoder<?>) loaded.reranker()).document(doc.text()).count();
    }

    // ---- harness ----

    static double timeMs(Runnable leg) {
        long t0 = System.nanoTime();
        leg.run();
        return (System.nanoTime() - t0) / 1e6;
    }

    static double median(double[] ms) {
        double[] sorted = ms.clone();
        Arrays.sort(sorted);
        return sorted[sorted.length / 2];
    }

    // ---- the /v1/rerank leg: same workload, measured by the same clock ----

    @SuppressWarnings("unchecked")
    static double[] httpLeg(String base, String query, List<TextSegment> docs) throws Exception {
        HttpClient client = HttpClient.newHttpClient();
        StringBuilder body = new StringBuilder("{\"query\":");
        body.append(Json.stringify(query)).append(",\"documents\":[");
        for (int i = 0; i < docs.size(); i++) {
            if (i > 0) body.append(',');
            body.append(Json.stringify(docs.get(i).text()));
        }
        body.append("]}");
        HttpRequest req =
                HttpRequest.newBuilder(URI.create(base + "/v1/rerank"))
                        .header("Content-Type", "application/json")
                        .POST(HttpRequest.BodyPublishers.ofString(body.toString()))
                        .build();
        client.send(req, HttpResponse.BodyHandlers.ofString()); // warmup
        client.send(req, HttpResponse.BodyHandlers.ofString());
        double[] ms = new double[3];
        String last = null;
        for (int r = 0; r < 3; r++) {
            long t0 = System.nanoTime();
            last = client.send(req, HttpResponse.BodyHandlers.ofString()).body();
            ms[r] = (System.nanoTime() - t0) / 1e6;
        }
        Arrays.sort(ms);
        Map<String, Object> parsed = (Map<String, Object>) Json.parse(last);
        List<Map<String, Object>> results = (List<Map<String, Object>>) parsed.get("results");
        double[] out = new double[1 + docs.size()];
        out[0] = ms[1];
        for (Map<String, Object> r : results) {
            int index = ((Number) r.get("index")).intValue();
            out[1 + index] = ((Number) r.get("relevance_score")).doubleValue();
        }
        return out;
    }

    static double spearman(List<Double> a, List<Double> b) {
        int n = a.size();
        double[] ra = ranks(a), rb = ranks(b);
        double d2 = 0;
        for (int i = 0; i < n; i++) d2 += (ra[i] - rb[i]) * (ra[i] - rb[i]);
        return 1 - 6 * d2 / (n * (double) (n * n - 1));
    }

    static double[] ranks(List<Double> v) {
        Integer[] idx = new Integer[v.size()];
        for (int i = 0; i < idx.length; i++) idx[i] = i;
        Arrays.sort(idx, (x, y) -> Double.compare(v.get(y), v.get(x)));
        double[] r = new double[v.size()];
        for (int rank = 0; rank < idx.length; rank++) r[idx[rank]] = rank;
        return r;
    }
}
