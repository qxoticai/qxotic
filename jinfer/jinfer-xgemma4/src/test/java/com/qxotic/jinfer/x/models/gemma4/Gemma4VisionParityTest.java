package com.qxotic.jinfer.x.models.gemma4;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.FloatTensor;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jinfer.x.PanamaMemoryArena;
import com.qxotic.jinfer.x.Views;
import com.qxotic.jinfer.x.boundary.Media;
import com.qxotic.jinfer.x.boundary.MediaProjector;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

class Gemma4VisionParityTest {
    private static final float TOLERANCE = 2e-3f;

    @ParameterizedTest(name = "{0}")
    @CsvSource({
        "gemma4v, hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf",
        "gemma4uv, hf.co/unsloth/gemma-4-12b-it-GGUF/mmproj-F32.gguf"
    })
    void matchesLegacyProjector(String type, String ref) throws Exception {
        Path path = TestModels.require(ref);
        float[] pixels = imagePixels(96, 48);
        var oldImage = new com.qxotic.jinfer.Media.Image(pixels, 48, 96, 3);
        var xImage = new Media.Image(pixels, 48, 96, 3);

        try (Arena oldArena = Arena.ofShared();
                Arena xArena = Arena.ofShared()) {
            FloatTensor expected;
            MediaProjector<Media.Image> actualProjector;
            if (type.equals("gemma4v")) {
                expected =
                        com.qxotic.jinfer.models.gemma4.Gemma4Vision.loadModel(path, oldArena)
                                .encode(oldImage);
                actualProjector = Gemma4Vision.loadModel(path, xArena);
            } else {
                expected =
                        com.qxotic.jinfer.models.gemma4.Gemma4VisionUnified.loadModel(
                                        path, oldArena)
                                .encode(oldImage);
                actualProjector = Gemma4VisionUnified.loadModel(path, xArena);
            }

            int rows = actualProjector.positions(xImage);
            actualProjector.project(
                    xImage,
                    rows,
                    actual -> assertClose(expected, Views.castToSegmentBacked(actual, "rows")));
        }
    }

    @Test
    void matchesLegacyDecoderAcrossBidirectionalImageBlock() throws Exception {
        Path text = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0");
        Path mmproj = TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/mmproj-F32.gguf");
        float[] pixels = imagePixels(96, 48);
        var oldImage = new com.qxotic.jinfer.Media.Image(pixels, 48, 96, 3);
        var xImage = new Media.Image(pixels, 48, 96, 3);

        try (Arena oldArena = Arena.ofShared();
                Arena xArena = Arena.ofShared()) {
            var old =
                    com.qxotic.jinfer.models.gemma4.Gemma4.loadModel(text, oldArena)
                            .attachMediaEncoders(mmproj, oldArena);
            var x = Gemma4.loadModel(text, mmproj, xArena);
            int[] prefix = x.tokenizer().encodeToArray("Look:");
            int rows = x.projector(Media.Image.class).orElseThrow().positions(xImage);
            assertTrue(rows > 1, "test image must exercise bidirectional attention");
            int capacity = prefix.length + rows + 2;
            try (var oldState = old.newState(capacity, rows);
                    var xState = x.newState(capacity, rows)) {
                old.ingest(oldState, com.qxotic.jinfer.Batch.prefill(prefix));
                x.ingest(xState, com.qxotic.jinfer.x.boundary.Batch.prefill(prefix));
                float[][] exactRows = new float[1][];
                old.embedder(com.qxotic.jinfer.Media.Image.class)
                        .orElseThrow()
                        .embed(
                                oldImage,
                                rows,
                                projected -> {
                                    exactRows[0] = new float[Math.toIntExact(projected.size())];
                                    for (int i = 0; i < exactRows[0].length; i++)
                                        exactRows[0][i] = projected.getFloat(i);
                                    old.ingest(
                                            oldState,
                                            com.qxotic.jinfer.Batch.embeddings(
                                                    projected, rows, true));
                                });
                MemoryView<MemorySegment> projected =
                        Views.allocateF32(
                                new PanamaMemoryArena(xArena),
                                rows,
                                x.configuration().embeddingLength());
                Views.copyFromArray(
                        projected, 0, exactRows[0], 0, exactRows[0].length, "projected rows");
                x.ingest(
                        xState,
                        com.qxotic.jinfer.x.boundary.Batch.embeddings(projected, rows, true));
                assertClose(
                        old.logits(oldState),
                        Views.castToSegmentBacked(x.logits(xState), "logits"));
            }
        }
    }

    private static void assertClose(FloatTensor expected, MemoryView<MemorySegment> actual) {
        int size = Math.toIntExact(actual.shape().size());
        assertEquals(expected.size(), size);
        float max = 0f;
        for (int i = 0; i < size; i++)
            max = Math.max(max, Math.abs(expected.getFloat(i) - Views.getFloat(actual, i, "rows")));
        assertTrue(max < TOLERANCE, "output diverged: max abs error " + max);
    }

    private static float[] imagePixels(int width, int height) {
        float[] pixels = new float[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++)
                    pixels[(y * width + x) * 3 + c] = ((x * 17 + y * 31 + c * 47) & 255) / 255f;
        return pixels;
    }
}
