package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.testkit.LastRowPrefill;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import java.util.Arrays;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Two trailing conv layers: the tail carries 1 + 2 x 2 rows past the last attention layer. */
@Tag("integration")
final class Lfm2PrunedPrefillIT {

    @Test
    void lastRowPrefillMatchesTheFullPass() throws Exception {
        Lfm2 model =
                Lfm2.loadModel(
                        TestModels.require("hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q8_0"), Arena.ofAuto());
        // layers 28 and 29 are conv layers of kernel 3: 1 + 2 x 2; logits barely see a short tail
        // (one conv tap), so pin the count
        assertEquals(5, model.configuration().tailRows());
        int[] text = model.tokenizer().encode(LastRowPrefill.COUNTING).toArray();
        // 6 rows skip one
        for (int length : new int[] {6, 37, 300, 700})
            LastRowPrefill.assertMatchesFullPass(model, Arrays.copyOf(text, length), 0.01);
    }
}
