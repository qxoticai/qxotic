package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The reranking laws on a real reranker GGUF (Qwen3-Reranker 0.6B). Assertions are ORDERING-based
 * or contract bounds, never absolute scores - relevance gaps dwarf the backend's warm FP jitter.
 * Model-gated: assume-skips when the file is absent.
 */
@Tag("integration")
class JinferScoringModelIT {

    static JinferScoringModel scorer;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(
                Files.exists(
                        TestModels.find(
                                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf"))),
                "model not found: "
                        + TestModels.find(
                                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")
                                .orElse(
                                        Path.of(
                                                "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")));
        scorer =
                JinferScoringModel.builder()
                        .modelPath(
                                TestModels.find(
                                                "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")
                                        .orElse(
                                                Path.of(
                                                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")))
                        .contextLength(2048)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (scorer != null) {
            scorer.close();
            scorer.close(); // idempotency pin: a second close must be a no-op, never ISE
        }
    }

    @Test
    void forkOfAnOwningModelRefusesWithTheRecipe() {
        IllegalStateException e =
                org.junit.jupiter.api.Assertions.assertThrows(
                        IllegalStateException.class, scorer::fork);
        org.junit.jupiter.api.Assertions.assertTrue(
                e.getMessage().contains("Models.loadReranker"), e.getMessage());
    }

    @Test
    void sharedWeightsForkScoresTheSameRanking() throws Exception {
        try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofShared()) {
            var loaded =
                    com.qxotic.jinfer.x.chat.Models.loadReranker(
                            TestModels.find(
                                            "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")
                                    .orElse(
                                            Path.of(
                                                    "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")),
                            arena);
            JinferScoringModel a =
                    JinferScoringModel.builder().model(loaded).contextLength(2048).build();
            JinferScoringModel b = a.fork();
            try {
                var docs =
                        java.util.List.of(
                                dev.langchain4j.data.segment.TextSegment.from(
                                        "The reset portal is https://acme.example/reset."),
                                dev.langchain4j.data.segment.TextSegment.from(
                                        "Bananas are rich in potassium."));
                var sa = a.scoreAll(docs, "Where do I reset my password?").content();
                var sb = b.scoreAll(docs, "Where do I reset my password?").content();
                org.junit.jupiter.api.Assertions.assertTrue(sa.get(0) > sa.get(1), sa.toString());
                org.junit.jupiter.api.Assertions.assertTrue(sb.get(0) > sb.get(1), sb.toString());
            } finally {
                a.close();
                b.close();
            }
        }
    }

    @Test
    void relevantDocumentOutranksDistractors() {
        List<TextSegment> docs =
                List.of(
                        TextSegment.from(
                                "The Eiffel Tower is a wrought-iron lattice tower in Paris,"
                                        + " completed in 1889 as the entrance to the World's"
                                        + " Fair."),
                        TextSegment.from(
                                "Photosynthesis converts light energy into chemical energy in"
                                        + " plants, producing oxygen as a byproduct."),
                        TextSegment.from(
                                "The recipe calls for two cups of flour, a pinch of salt and"
                                        + " three eggs, whisked until smooth."));
        Response<List<Double>> r = scorer.scoreAll(docs, "When was the Eiffel Tower built?");
        List<Double> scores = r.content();
        assertEquals(3, scores.size(), "one score per segment, input order");
        assertTrue(
                scores.get(0) > scores.get(1) && scores.get(0) > scores.get(2),
                "relevant document must outrank both distractors: " + scores);
        assertTrue(r.tokenUsage().inputTokenCount() > 0, "prompt tokens billed");
    }

    @Test
    void frameRewindLeavesNoResidue() {
        // the shared-prefix reuse law: the same (query, document) pair scored first and last
        // in one batch must agree - the cursor rewind may leave nothing behind (tolerance
        // covers scheduling-order FP jitter, orders of magnitude below any relevance gap)
        TextSegment doc = TextSegment.from("The Eiffel Tower is a lattice tower in Paris.");
        TextSegment other = TextSegment.from("Photosynthesis happens in plant chloroplasts.");
        List<Double> scores =
                scorer.scoreAll(List.of(doc, other, doc), "Where is the Eiffel Tower?").content();
        assertTrue(
                Math.abs(scores.get(0) - scores.get(2)) < 1e-3,
                "rewound pair must score like the fresh one: " + scores);
    }

    @Test
    void scoresAreProbabilities() {
        // the documented contract: [0,1], higher is more relevant - consumers threshold on it
        // (langchain4j's ReRankingContentAggregator.minScore gates the whole RAG context this way)
        List<Double> scores =
                scorer.scoreAll(
                                List.of(
                                        TextSegment.from("The Eiffel Tower stands in Paris."),
                                        TextSegment.from(
                                                "Bread dough needs to rest before" + " baking.")),
                                "Where is the Eiffel Tower?")
                        .content();
        for (double score : scores) {
            assertTrue(score >= 0.0 && score <= 1.0, "score out of [0,1]: " + score);
        }
    }

    @Test
    void nothingToScoreIsNotAnError() {
        // a retriever that found nothing must cost nothing - not a crash, not a wasted prefill
        Response<List<Double>> empty = scorer.scoreAll(List.of(), "anything at all");
        assertTrue(empty.content().isEmpty());
        assertEquals(0, empty.tokenUsage().inputTokenCount(), "no candidates, no tokens billed");
    }

    @Test
    void documentOverTheContextIsRefusedByIndex() {
        TextSegment small = TextSegment.from("The Eiffel Tower is in Paris.");
        TextSegment huge = TextSegment.from("lattice tower ".repeat(2000)); // way past 2048 tokens
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> scorer.scoreAll(List.of(small, huge), "Where is the Eiffel Tower?"));
        assertTrue(e.getMessage().contains("document 1"), e.getMessage());
        assertTrue(e.getMessage().contains("contextLength"), e.getMessage());
    }

    @Test
    void aChatModelIsNotAReranker() {
        // the mistake to expect: pointing the scorer at the chat GGUF already on disk. LFM2 IS
        // a reranker family (ColBERT), so its provider refuses the wrong CHECKPOINT by name and
        // points at the right one - before any weight is mapped
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                JinferScoringModel.builder()
                                        .modelPath(
                                                TestModels.require(
                                                        "hf.co/LiquidAI/LFM2.5-350M-GGUF/LFM2.5-350M-Q8_0.gguf"))
                                        .build());
        assertTrue(e.getMessage().contains("not the family's reranker"), e.getMessage());
        assertTrue(e.getMessage().contains("LFM2.5-ColBERT"), e.getMessage());
    }

    @Test
    void useAfterCloseFailsLoudly() {
        JinferScoringModel closed =
                JinferScoringModel.builder()
                        .modelPath(
                                TestModels.find(
                                                "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")
                                        .orElse(
                                                Path.of(
                                                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf")))
                        .contextLength(512)
                        .build();
        closed.close();
        assertThrows(
                IllegalStateException.class,
                () -> closed.scoreAll(List.of(TextSegment.from("anything")), "a query"),
                "scoring a closed model must fail, never read freed memory");
    }

    @Test
    void rankingIsQuerySensitive() {
        // the same corpus reranks differently under a different question - scores are a
        // function of the PAIR, not of the documents alone (the whole point of a reranker)
        List<TextSegment> docs =
                List.of(
                        TextSegment.from(
                                "The Eiffel Tower is a wrought-iron lattice tower in Paris."),
                        TextSegment.from(
                                "Photosynthesis converts light energy into chemical energy in"
                                        + " plants."));
        List<Double> forTower = scorer.scoreAll(docs, "Where is the Eiffel Tower?").content();
        List<Double> forPlants =
                scorer.scoreAll(docs, "How do plants convert sunlight to energy?").content();
        assertTrue(forTower.get(0) > forTower.get(1), "tower query prefers doc 0: " + forTower);
        assertTrue(forPlants.get(1) > forPlants.get(0), "plant query prefers doc 1: " + forPlants);
    }
}
