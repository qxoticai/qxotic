package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.lang.foreign.Arena;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM2.5-Embedding parity against llama.cpp (b-source of the golden CSV: {@code llama-embedding
 * --embd-output-format json --embd-normalize 2} on the same Q8_0 GGUF, converted to one
 * comma-separated vector per line). The embedder is bidirectional with a CENTERED short-conv window
 * - both easy to get subtly wrong in a way retrieval-quality checks would never catch, so the
 * assertion is per-VECTOR cosine against the reference stack, not a ranking.
 */
@Tag("integration")
class Lfm2EmbedParityIT {

    /** Must match the texts the golden CSV was generated from, in order. */
    private static final List<String> TEXTS =
            List.of(
                    "query: What is panda?",
                    "document: The giant panda (Ailuropoda melanoleuca), sometimes called a panda"
                            + " bear or simply panda, is a bear species endemic to China.",
                    "document: it is a bear",
                    "document: hi",
                    "query: Wie funktioniert ein Kernkraftwerk?",
                    "document: A nuclear power plant heats water into steam that drives a turbine"
                            + " connected to a generator.");

    @Test
    void matchesLlamaCppEmbeddings() throws Exception {
        List<double[]> golden = golden();
        assertEquals(TEXTS.size(), golden.size(), "golden CSV rows");
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> embedder =
                    Models.loadEmbedder(ModelFixture.LFM25_EMBEDDING_350M_Q8.require(), arena);
            assertEquals(1024, embedder.dimension());

            // a TINY batch capacity forces multi-group embedding, so the group-boundary and
            // CLS-row-within-a-later-group paths are exercised, not just the one-group case
            var state = embedder.model().newState(2048, 64, arena);
            List<double[]> vectors = new ArrayList<>();
            embedder.embedAll(
                    state,
                    2048,
                    TEXTS,
                    v -> {
                        double[] d = new double[embedder.dimension()];
                        for (int i = 0; i < d.length; i++) d[i] = v.getFloat(i);
                        vectors.add(d);
                    });
            for (int i = 0; i < TEXTS.size(); i++) {
                double cos = cosine(vectors.get(i), golden.get(i));
                assertTrue(cos > 0.999, "vector " + i + " diverged from llama.cpp: cos=" + cos);
            }
            // and the embeddings are L2-normalized, as every consumer assumes
            for (double[] v : vectors) {
                double norm = Math.sqrt(Arrays.stream(v).map(x -> x * x).sum());
                assertEquals(1.0, norm, 1e-3, "not L2-normalized");
            }
        }
    }

    @Test
    void aSequenceOverTheBatchIsRefusedByName() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> embedder =
                    Models.loadEmbedder(ModelFixture.LFM25_EMBEDDING_350M_Q8.require(), arena);
            var state = embedder.model().newState(2048, 16, arena);
            // bidirectional attention forwards a sequence whole, so one over the cap is a clear
            // refusal, never a silently-truncated embedding
            String big = "word ".repeat(64);
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> embedder.embedAll(state, 2048, List.of(big), v -> {}));
            assertTrue(e.getMessage().contains("whole"), e.getMessage());
        }
    }

    @Test
    void aGenerativeCheckpointIsRefusedAsAnEmbedder() {
        try (Arena arena = Arena.ofShared()) {
            IllegalArgumentException e =
                    assertThrows(
                            IllegalArgumentException.class,
                            () -> Models.loadEmbedder(ModelFixture.LFM25_350M_Q8.require(), arena));
            assertTrue(e.getMessage().contains("generative"), e.getMessage());
            assertTrue(e.getMessage().contains("LFM2.5-Embedding"), e.getMessage());
        }
    }

    private static List<double[]> golden() throws Exception {
        List<double[]> vectors = new ArrayList<>();
        try (BufferedReader in =
                new BufferedReader(
                        new InputStreamReader(
                                Lfm2EmbedParityIT.class.getResourceAsStream(
                                        "/lfm25-embedding-350m-q8-golden.csv"),
                                StandardCharsets.UTF_8))) {
            for (String line; (line = in.readLine()) != null; ) {
                if (line.isBlank()) continue;
                vectors.add(
                        Arrays.stream(line.split(",")).mapToDouble(Double::parseDouble).toArray());
            }
        }
        return vectors;
    }

    private static double cosine(double[] a, double[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        return dot / Math.sqrt(na * nb);
    }
}
