package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import java.nio.file.Files;
import java.util.List;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The reranking laws on a real reranker GGUF (Qwen3-Reranker 0.6B). Assertions are ORDERING-based,
 * never absolute scores - relevance gaps dwarf the backend's warm FP jitter. Model-gated:
 * assume-skips when the file is absent.
 */
@Tag("integration")
class JinferScoringModelIT {

    static JinferScoringModel scorer;

    @BeforeAll
    static void load() {
        Assumptions.assumeTrue(
                Files.exists(ModelFixture.QWEN3_RERANKER_06B_Q8.path()),
                "model not found: " + ModelFixture.QWEN3_RERANKER_06B_Q8.path());
        scorer =
                JinferScoringModel.builder()
                        .modelPath(ModelFixture.QWEN3_RERANKER_06B_Q8.path())
                        .contextLength(2048)
                        .build();
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
