package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Media;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class Gemma4ConformerParityTest {
    @Test
    void matchesLegacyFinalEmbedding() throws Exception {
        Path path = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        float[] pcm = new float[10_000];
        for (int i = 0; i < pcm.length; i++) pcm[i] = (float) Math.sin(i * 0.07);

        try (Arena oldArena = Arena.ofShared();
                Arena xArena = Arena.ofShared()) {
            FloatTensor expected =
                    com.qxotic.jinfer.models.gemma4.Gemma4Conformer.loadModel(path, oldArena)
                            .encode(new com.qxotic.jinfer.Media.Audio(pcm, 16_000, 1));
            Gemma4Conformer actual = Gemma4Conformer.loadModel(path, xArena);
            int[] offset = {0};
            actual.project(
                    new Media.Audio(pcm, 16_000, 1),
                    2,
                    rows -> {
                        var view = Views.castToSegmentBacked(rows, "conformer output");
                        float maxError = 0f;
                        for (int i = 0; i < view.shape().size(); i++)
                            maxError =
                                    Math.max(
                                            maxError,
                                            Math.abs(
                                                    expected.getFloat(offset[0] + i)
                                                            - Views.getFloat(
                                                                    view, i, "conformer output")));
                        assertTrue(maxError < 2e-3f, "maximum absolute error: " + maxError);
                        offset[0] += Math.toIntExact(view.shape().size());
                    });
            assertEquals(expected.size(), offset[0]);
        }
    }
}
