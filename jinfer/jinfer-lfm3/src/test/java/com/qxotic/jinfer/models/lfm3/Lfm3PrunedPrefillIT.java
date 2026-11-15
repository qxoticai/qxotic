package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.testkit.LastRowPrefill;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** One trailing conv layer of kernel 7: the tail carries 7 rows past the last attention layer. */
@Tag("integration")
final class Lfm3PrunedPrefillIT {

    @Test
    void lastRowPrefillMatchesTheFullPass() throws Exception {
        Lfm3 model =
                Lfm3.loadModel(
                        TestModels.require("hf.co/LiquidAI/LFM3-1.7B-GGUF/LFM3-1.7B-Q8_0.gguf"),
                        Arena.ofAuto());
        // layer 23 is a conv layer of kernel 7: 1 + 6; logits barely see a short tail (one conv
        // tap), so pin the count
        assertEquals(7, model.configuration().tailRows());
        int[] text = model.tokenizer().encode(LastRowPrefill.COUNTING).toArray();
        // 8 rows skip one
        for (int length : new int[] {8, 37, 300, 700})
            LastRowPrefill.assertMatchesFullPass(model, Arrays.copyOf(text, length), 0.01);
    }
}
