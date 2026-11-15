package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.testkit.LastRowPrefill;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryAllocators;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The KV-shared layers run on the last row alone, as a decode step at its position: it reads every
 * other row's keys from the cache, as a next batch would, so logits move ~0.003 (F16 rounding, as a
 * split prompt's do); a wrong key or value row for the live token moves them by 3.5 to 12.
 */
@Tag("integration")
final class Gemma4PrunedPrefillIT {

    private static final String MODEL = "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0";

    @Test
    void lastRowPrefillMatchesTheFullPass() throws Exception {
        Gemma4 model = Gemma4.loadModel(TestModels.require(MODEL), Arena.ofAuto());
        int[] text = model.tokenizer().encode(LastRowPrefill.COUNTING).toArray();
        // 2 rows skip one; 700 wrap the 512-row sliding window ring
        for (int length : new int[] {2, 37, 700})
            LastRowPrefill.assertMatchesFullPass(model, Arrays.copyOf(text, length), 0.01);
    }

    @Test
    void lastRowMediaBlockMatchesTheFullPass() throws Exception {
        Gemma4 model =
                Gemma4.loadModel(
                        TestModels.require(MODEL),
                        TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf"),
                        Arena.ofAuto());
        // a bidirectional block of projected rows, as an image arrives: its last row sees them all
        int rows = 300, width = model.configuration().embeddingLength();
        float[] values = new float[rows * width];
        Random random = new Random(7);
        for (int i = 0; i < values.length; i++) values[i] = (float) random.nextGaussian();
        MemoryView<MemorySegment> block =
                Views.allocateF32(MemoryAllocators.ofArena(Arena.ofAuto()), rows, width);
        Views.copyFromArray(block, 0, values, 0, values.length, "block");
        LastRowPrefill.assertMatchesFullPass(
                model,
                outputs ->
                        List.of(new Batch(new Batch.Input.Embeddings(block, rows, true), outputs)),
                rows,
                0.01);
    }
}
