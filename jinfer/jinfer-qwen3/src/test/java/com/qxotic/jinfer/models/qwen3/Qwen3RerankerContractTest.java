package com.qxotic.jinfer.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.boundary.Batch;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Framing and chunking contracts for the Qwen3 cross-encoder recipe. */
@Tag("integration")
class Qwen3RerankerContractTest {

    private static final String REF = "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF:Q8_0";
    private static final String QUERY = "When was the Eiffel Tower built?";
    private static final String DOCUMENT = "The Eiffel Tower is a lattice tower in Paris.";
    private static final List<String> DOCUMENTS =
            List.of(
                    "The Eiffel Tower was completed in 1889 in Paris.",
                    "Bananas are a good source of potassium.",
                    "Gustave Eiffel's company built the tower between 1887 and 1889.");

    private static final int[] HEAD = {
        151644, 8948, 198, 60256, 3425, 279, 11789, 20027, 279, 8502, 3118, 389, 279, 11361, 323,
        279, 758, 1235, 3897, 13, 7036, 429, 279, 4226, 646, 1172, 387, 330, 9693, 1, 476, 330,
        2152, 3263, 151645, 198, 151644, 872, 198, 27, 641, 1235, 26818, 16246, 264, 3482, 2711,
        3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239, 198, 27, 2859, 26818, 3197, 572, 279,
        468, 3092, 301, 21938, 5798, 5267, 75692, 26818
    };
    private static final int[] DOCUMENT_IDS = {
        576, 468, 3092, 301, 21938, 374, 264, 54272, 21271, 304, 12095, 13, 151645, 198, 151644,
        77091, 198, 151667, 271, 151668, 271
    };

    @Test
    void frameIsTokenExact() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            Qwen3Reranker reranker = reranker(weights);
            assertArrayEquals(HEAD, ids(reranker.prefix(reranker.defaultInstruction(), QUERY)));
            assertArrayEquals(DOCUMENT_IDS, ids(reranker.document(DOCUMENT)));
        }
    }

    @Test
    void chunkingDoesNotChangeScoresOrTokenAccounting() throws Exception {
        try (Arena weights = Arena.ofShared()) {
            Qwen3Reranker reranker = reranker(weights);
            Scores whole = scores(reranker, 512);
            Scores chunked = scores(reranker, 16);
            int expectedTokens = reranker.prefix(reranker.defaultInstruction(), QUERY).count();
            for (String document : DOCUMENTS) expectedTokens += reranker.document(document).count();

            assertEquals(expectedTokens, whole.tokens());
            assertEquals(whole.tokens(), chunked.tokens());
            assertEquals(whole.values().size(), chunked.values().size());
            for (int i = 0; i < whole.values().size(); i++) {
                assertEquals(whole.values().get(i), chunked.values().get(i), 5e-3, "document " + i);
            }
        }
    }

    private static Qwen3Reranker reranker(Arena weights) throws Exception {
        Path path = TestModels.require(REF);
        return new Qwen3Reranker(Qwen3.loadModel(path, weights));
    }

    private static Scores scores(Qwen3Reranker reranker, int batchCapacity) {
        List<Double> values = new ArrayList<>();
        try (Qwen3.State state = reranker.model().newState(4096, batchCapacity)) {
            int tokens =
                    reranker.scoreAll(
                            state, reranker.defaultInstruction(), QUERY, DOCUMENTS, values::add);
            return new Scores(tokens, values);
        }
    }

    private static int[] ids(Batch batch) {
        return ((Batch.Input.Tokens) batch.input()).ids();
    }

    private record Scores(int tokens, List<Double> values) {}
}
