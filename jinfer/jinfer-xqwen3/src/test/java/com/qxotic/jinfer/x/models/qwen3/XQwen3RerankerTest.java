package com.qxotic.jinfer.x.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.x.boundary.Batch;
import java.io.IOException;
import java.lang.foreign.Arena;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

/**
 * The Qwen3Reranker port's gate, two halves of ONE convention. The frame leg pins the judge frame
 * token-exact against the ids captured from the shipped scorer (the old tree's
 * Qwen3RerankerFrameTest oracle): a framing edit that looks harmless moves every score. The score
 * leg runs the full frame-once-rewind-per-candidate loop on the REAL Qwen3-Reranker-0.6B Q8_0
 * checkpoint, old shipped scorer (via {@code Models.loadReranker}) vs the x port, at a batch
 * capacity that fits the whole frame and at one that forces chunking mid-frame (exercising the KV
 * carry AND the per-candidate cursor rewind) — token-count equality plus per-document score parity.
 * Both trees route gemm/gemv to the same dot-based Java floor (JAM surefire-excluded). Skipped when
 * the checkpoint is not in the HF cache.
 */
class XQwen3RerankerTest {

    private static final Path HF_CACHE =
            Path.of(System.getProperty("user.home"), ".cache/huggingface/hub");
    private static final double SCORE_TOLERANCE = 5e-3;

    private static final String QUERY = "When was the Eiffel Tower built?";
    private static final String DOCUMENT = "The Eiffel Tower is a lattice tower in Paris.";

    // <|im_start|>system\nJudge whether ... "yes" or "no".<|im_end|>\n<|im_start|>user\n
    // <Instruct>: {card default}\n<Query>: {QUERY}\n<Document>:
    private static final int[] HEAD = {
        151644, 8948, 198, 60256, 3425, 279, 11789, 20027, 279, 8502, 3118, 389, 279, 11361, 323,
        279, 758, 1235, 3897, 13, 7036, 429, 279, 4226, 646, 1172, 387, 330, 9693, 1, 476, 330,
        2152, 3263, 151645, 198, 151644, 872, 198, 27, 641, 1235, 26818, 16246, 264, 3482, 2711,
        3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239, 198, 27, 2859, 26818, 3197, 572, 279,
        468, 3092, 301, 21938, 5798, 5267, 75692, 26818
    };

    // " {DOCUMENT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    private static final int[] DOCUMENT_IDS = {
        576, 468, 3092, 301, 21938, 374, 264, 54272, 21271, 304, 12095, 13, 151645, 198, 151644,
        77091, 198, 151667, 271, 151668, 271
    };

    private static Path rerankerModel;

    @BeforeAll
    static void findModels() throws IOException {
        rerankerModel =
                findGguf(
                        "models--mradermacher--Qwen3-Reranker-0.6B-GGUF",
                        "Qwen3-Reranker-0.6B.Q8_0.gguf");
    }

    private static Path findGguf(String repoName, String fileName) throws IOException {
        Path repo = HF_CACHE.resolve(repoName).resolve("snapshots");
        if (!Files.isDirectory(repo)) return null;
        try (Stream<Path> snaps = Files.list(repo)) {
            return snaps.flatMap(
                            s -> {
                                try {
                                    return Files.list(s);
                                } catch (IOException e) {
                                    return Stream.empty();
                                }
                            })
                    .filter(p -> p.getFileName().toString().equals(fileName))
                    .findFirst()
                    .orElse(null);
        }
    }

    @Test
    void frameIsTokenExact() throws Exception {
        assumeTrue(rerankerModel != null, "Qwen3-Reranker-0.6B.Q8_0.gguf not in the HF cache");
        Qwen3 xm = Qwen3.loadModel(rerankerModel, Arena.ofAuto());
        Qwen3Reranker reranker = new Qwen3Reranker(xm);
        // the reuse law rides in these two arrays: the frame ends AT the document opener (id 26818,
        // the ':' of "<Document>:") and the candidate carries the leading space, so the split is a
        // prefix cut, not a re-tokenization - the two halves concatenate to the joint encoding
        assertArrayEquals(
                HEAD, ids(reranker.head(reranker.defaultInstruction(), QUERY)), "judge frame");
        assertArrayEquals(DOCUMENT_IDS, ids(reranker.document(DOCUMENT)), "document framing");
    }

    @Test
    void scoreParityWholeFrame() throws Exception {
        assumeTrue(rerankerModel != null, "Qwen3-Reranker-0.6B.Q8_0.gguf not in the HF cache");
        assertScoreParity(512); // frame + document fit in one chunk
    }

    @Test
    void scoreParityChunked() throws Exception {
        assumeTrue(rerankerModel != null, "Qwen3-Reranker-0.6B.Q8_0.gguf not in the HF cache");
        assertScoreParity(16); // chunks mid-frame AND mid-document
    }

    private static final List<String> DOCUMENTS =
            List.of(
                    "The Eiffel Tower was completed in 1889 for the Exposition Universelle in"
                            + " Paris.",
                    "Bananas are a good source of potassium.",
                    "Gustave Eiffel's company designed and built the tower between 1887 and 1889;"
                            + " it was initially criticized by Parisian artists.");

    private static void assertScoreParity(int batchCapacity) throws Exception {
        LoadedReranker<?> loaded = Models.loadReranker(rerankerModel, Arena.ofAuto());
        Qwen3 xm = Qwen3.loadModel(rerankerModel, Arena.ofAuto());
        Qwen3Reranker xr = new Qwen3Reranker(xm);
        String instruction = xr.defaultInstruction();

        List<Double> oldScores = new ArrayList<>();
        com.qxotic.jinfer.RuntimeState oldState = loaded.model().newState(4096, batchCapacity);
        int oldTokens = loaded.scoreAll(oldState, instruction, QUERY, DOCUMENTS, oldScores::add);

        List<Double> xScores = new ArrayList<>();
        Qwen3.State xState = xm.newState(4096, batchCapacity);
        int xTokens = xr.scoreAll(xState, instruction, QUERY, DOCUMENTS, xScores::add);

        assertEquals(oldTokens, xTokens, "token total");
        assertEquals(oldScores.size(), xScores.size(), "score count");
        for (int i = 0; i < oldScores.size(); i++) {
            double o = oldScores.get(i), x = xScores.get(i);
            assertTrue(
                    Math.abs(o - x) < SCORE_TOLERANCE,
                    "doc " + i + ": old=" + o + " x=" + x + " (bc=" + batchCapacity + ")");
        }
        // the convention works at all: the two Eiffel docs outscore the banana
        assertTrue(xScores.get(0) > xScores.get(1), "relevant doc 0 beats the decoy");
        assertTrue(xScores.get(2) > xScores.get(1), "relevant doc 2 beats the decoy");
    }

    private static int[] ids(Batch batch) {
        return ((Batch.Input.Tokens) batch.input()).ids();
    }
}
