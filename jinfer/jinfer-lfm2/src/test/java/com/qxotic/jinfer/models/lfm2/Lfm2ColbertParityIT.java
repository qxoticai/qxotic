package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.lang.foreign.Arena;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * LFM2.5-ColBERT reranking parity against llama.cpp: golden MaxSim scores computed by the reference
 * flow ({@code llama-server --embeddings} per-token embeddings on the same Q8_0 GGUF,
 * L2-normalized, MaxSim with the model card's preprocessing - {@code [Q] }/{@code [D] } marker
 * TOKENS, query padded to 32, punctuation skiplist on document rows). The recipe has three
 * hand-offs a ranking test would never catch drifting - the marker tokens, the pad expansion rows,
 * the skiplist - so the assertion is absolute scores, not order.
 */
@Tag("integration")
class Lfm2ColbertParityIT {

    private static final String QUERY = "What is panda?";
    private static final List<String> DOCUMENTS =
            List.of(
                    "hi",
                    "it is a bear",
                    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or"
                            + " simply panda, is a bear species endemic to China.");

    /** llama.cpp reference scores for (QUERY, DOCUMENTS) on this exact Q8_0 file. */
    private static final double[] GOLDEN_SCORES = {28.9881, 29.5442, 30.0316};

    @Test
    void matchesLlamaCppMaxSimScores() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedReranker<?> reranker =
                    Models.loadReranker(ModelFixture.LFM25_COLBERT_350M_Q8.require(), arena);
            var state = reranker.model().newState(2048, 512, arena);
            List<Double> scores = new ArrayList<>();
            reranker.scoreAll(state, "", QUERY, DOCUMENTS, scores::add);
            assertEquals(DOCUMENTS.size(), scores.size());
            for (int i = 0; i < GOLDEN_SCORES.length; i++) {
                assertEquals(
                        GOLDEN_SCORES[i],
                        scores.get(i),
                        0.05, // quant kernel-order noise; measured deltas are <= 0.03
                        "document " + i + " drifted from llama.cpp");
            }
            // the point of a reranker: the on-topic document wins
            assertTrue(scores.get(2) > scores.get(1) && scores.get(1) > scores.get(0));
        }
    }

    @Test
    void anInstructionIsRefusedByName() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedReranker<?> reranker =
                    Models.loadReranker(ModelFixture.LFM25_COLBERT_350M_Q8.require(), arena);
            var state = reranker.model().newState(2048, 512, arena);
            // MaxSim has no instruction slot; silently ignoring one would misreport what ran
            UnsupportedOperationException e =
                    assertThrows(
                            UnsupportedOperationException.class,
                            () ->
                                    reranker.scoreAll(
                                            state, "Judge relevance.", QUERY, DOCUMENTS, s -> {}));
            assertTrue(e.getMessage().contains("instruction"), e.getMessage());
        }
    }

    @Test
    void theWrongCheckpointsRefuseByName() {
        try (Arena arena = Arena.ofShared()) {
            if (ModelFixture.LFM25_350M_Q8.present()) {
                // a generative LFM2 is not the family's reranker
                IllegalArgumentException e =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        Models.loadReranker(
                                                ModelFixture.LFM25_350M_Q8.path(), arena));
                assertTrue(e.getMessage().contains("ColBERT"), e.getMessage());
            }
            if (ModelFixture.LFM25_COLBERT_350M_Q8.present()) {
                // and a retrieval checkpoint must never "chat" - it would generate noise
                IllegalArgumentException e =
                        assertThrows(
                                IllegalArgumentException.class,
                                () ->
                                        Models.load(
                                                ModelFixture.LFM25_COLBERT_350M_Q8.path(), arena));
                assertTrue(e.getMessage().contains("RETRIEVAL"), e.getMessage());
            }
        }
    }
}
