package com.qxotic.jinfer.models.qwen3;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.chat.LoadedReranker;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.chat.Reranker;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.io.IOException;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The judge frame is a wire format: the model card's exact bytes, tokenized in the card's exact
 * domains (scaffold as trusted ids, instruction/query/document plain). These ids were captured from
 * the shipped scorer and must not drift - a frame edit that looks harmless moves every score.
 *
 * <p>Goes through {@link Models#loadReranker}, so it also pins the architecture dispatch and the
 * verdict-token resolution that loading performs.
 */
@Tag("integration")
class Qwen3RerankerFrameTest {

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

    @Test
    void frameIsTokenExact() throws IOException {
        LoadedReranker<?> loaded =
                Models.loadReranker(
                        ModelFixture.QWEN3_RERANKER_06B_Q8.require(), 512, Arena.ofAuto());
        Reranker<?> reranker = loaded.reranker();
        Batch head = reranker.head(reranker.defaultInstruction(), QUERY);
        Batch document = reranker.document(DOCUMENT);
        // the reuse law rides in these two arrays: the frame ends AT the document opener (id 26818,
        // the ':' of "<Document>:") and the candidate carries the leading space, so the split is a
        // prefix cut, not a re-tokenization - the two halves concatenate to the joint encoding
        assertArrayEquals(HEAD, ids(head), "judge frame drifted");
        assertArrayEquals(DOCUMENT_IDS, ids(document), "document framing drifted");
    }

    private static int[] ids(Batch batch) {
        return ((Batch.Input.Tokens) batch.input()).ids();
    }
}
