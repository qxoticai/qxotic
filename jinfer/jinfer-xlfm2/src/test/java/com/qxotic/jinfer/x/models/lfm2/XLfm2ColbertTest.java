package com.qxotic.jinfer.x.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.format.gguf.GGUF;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.boundary.Reranker;
import com.qxotic.jinfer.x.chat.Models;
import com.qxotic.jinfer.x.kernels.ModelLoader;
import com.qxotic.toknroll.Tokenizer;
import com.qxotic.toknroll.gguf.GGUFTokenizerLoader;
import java.lang.foreign.Arena;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * LFM2.5-ColBERT-350M-Q8_0 reranking parity against llama.cpp. The golden MaxSim scores come from
 * {@code llama-server --embeddings} per-token embeddings on this exact Q8_0 file, L2-normalized,
 * with the model card's preprocessing. The absolute scores catch drift in the three hand-offs
 * (marker tokens, pad expansion rows, skiplist) that a ranking-only test would miss. Skipped when
 * the checkpoint is not cached.
 */
class XLfm2ColbertTest {

    private static final String QUERY = "What is panda?";
    private static final List<String> DOCUMENTS =
            List.of(
                    "hi",
                    "it is a bear",
                    "The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or"
                            + " simply panda, is a bear species endemic to China.");

    /** llama.cpp reference scores for (QUERY, DOCUMENTS) on this exact Q8_0 file. */
    private static final double[] GOLDEN_SCORES = {28.9881, 29.5442, 30.0316};

    private static Path model() {
        return TestModels.require("hf.co/LiquidAI/LFM2.5-ColBERT-350M-GGUF:Q8_0");
    }

    @Test
    void matchesLlamaCppMaxSimScores() throws Exception {
        Path path = model();
        Tokenizer tokenizer;
        try (FileChannel channel = FileChannel.open(path)) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5-colbert");
            tokenizer = GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
        }
        var loaded = Models.loadReranker(path, Arena.ofAuto(), tokenizer);
        assertSame(tokenizer, ((Lfm2) loaded.model()).tokenizer());
        try (var state = loaded.model().newState(2048, 512)) {
            List<Double> scores = new ArrayList<>();
            loaded.scoreAll(state, "", QUERY, DOCUMENTS, scores::add);
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
        try (FileChannel channel = FileChannel.open(model())) {
            GGUF gguf = ModelLoader.readGguf(channel, "lfm2.5-colbert");
            Tokenizer tokenizer =
                    GGUFTokenizerLoader.createBuilderWithBuiltins().build().fromGGUF(gguf);
            var xm = Lfm2.loadModel(channel, gguf, Arena.ofAuto(), tokenizer);
            Reranker<Lfm2.State> reranker = Lfm2Colbert.fromGguf(xm, gguf);
            try (var state = xm.newState(2048, 512)) {
                // MaxSim has no instruction slot; silently ignoring one would misreport what ran
                UnsupportedOperationException e =
                        assertThrows(
                                UnsupportedOperationException.class,
                                () ->
                                        reranker.scoreAll(
                                                state,
                                                "Judge relevance.",
                                                QUERY,
                                                DOCUMENTS,
                                                s -> {}));
                assertTrue(e.getMessage().contains("instruction"), e.getMessage());
            }
        }
    }

    @Test
    void aNonColbertCheckpointIsRefusedBeforeTokenizerUse() {
        Lfm2.Configuration config =
                new Lfm2.Configuration(
                        4,
                        new int[] {8},
                        1,
                        1,
                        new int[] {1},
                        16,
                        8,
                        1e-5f,
                        10_000f,
                        4,
                        0,
                        3,
                        0,
                        0,
                        0,
                        1,
                        1,
                        true,
                        0,
                        0);
        Lfm2 model = new Lfm2(config, null, new Lfm2.Weights(null, null, null, null, null, null));

        IllegalArgumentException error =
                assertThrows(IllegalArgumentException.class, () -> new Lfm2Colbert(model, 1, 0));
        assertTrue(error.getMessage().contains("ColBERT"), error.getMessage());
    }
}
