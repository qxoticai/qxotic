package com.qxotic.jinfer.models.lfm2;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.Views;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.media.Multimodal;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryView;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class Lfm2VisionModelTest {
    private static final String TEXT = "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF:Q4_K_M";
    private static final String PROJECTOR =
            "hf.co/LiquidAI/LFM2.5-VL-3B-GGUF/mmproj-LFM2.5-VL-3B-Q8_0.gguf";

    @Test
    void loadsOfficialQ8ProjectorAndProducesFiniteBorrowedRows() throws Exception {
        Path path = TestModels.require(PROJECTOR);
        List<MemorySegment> borrowed = new ArrayList<>();
        try (Arena arena = Arena.ofShared()) {
            Lfm2Vision vision = Lfm2Vision.loadModel(path, arena);
            Media.Image image = shiftedRainbow(256);
            assertEquals(64, vision.positions(image));
            vision.project(
                    image,
                    64,
                    rows -> {
                        assertEquals(64L, rows.shape().flatAt(0));
                        assertEquals(2048L, rows.shape().flatAt(1));
                        MemoryView<MemorySegment> segmentRows =
                                Views.castToSegmentBacked(rows, "rows");
                        MemorySegment segment = (MemorySegment) segmentRows.memory().base();
                        assertTrue(segment.scope().isAlive());
                        float[] values = Views.toFloatArray(segmentRows, "rows");
                        float sum = 0;
                        for (float value : values) {
                            assertTrue(Float.isFinite(value));
                            sum += value;
                        }
                        // Fresh llama.cpp, same Q8 projector, raw 256px rainbow input. The Java
                        // image is shifted because this API owns the model's x*2-1 normalization.
                        assertEquals(215.128174f, sum, 5f);
                        assertArrayEquals(
                                new float[] {0.0302f, -0.0598f, 0.2365f},
                                Arrays.copyOf(values, 3),
                                0.01f);
                        assertArrayEquals(
                                new float[] {0.1229f, -0.0922f, -0.1178f},
                                Arrays.copyOfRange(values, 2045, 2048),
                                0.01f);
                        borrowed.add(segment);
                    });
        }
        assertFalse(borrowed.getFirst().scope().isAlive());
    }

    @Test
    void publicProviderAttachesSidecarAndIngestsImageRows() throws Exception {
        Path text = TestModels.require(TEXT);
        Path projector = TestModels.require(PROJECTOR);
        try (Arena weights = Arena.ofShared()) {
            var loaded = Models.load(text, weights, Map.of("media", projector));
            Lfm2 model = assertInstanceOf(Lfm2.class, loaded.model());
            Multimodal media = assertInstanceOf(Multimodal.class, model);
            assertTrue(media.projector(Media.Image.class).isPresent());
            assertTrue(loaded.template().isPresent());

            Media.Image image = new Media.Image(new float[64 * 64 * 3], 64, 64, 3);
            int[] prefix = loaded.tokenizer().encodeToArray("Look:");
            try (Lfm2.State state = model.newState(128, 64)) {
                model.ingest(state, Batch.prefill(prefix));
                media.projector(Media.Image.class)
                        .orElseThrow()
                        .project(
                                image,
                                64,
                                rows -> model.ingest(state, Batch.embeddings(rows, 64, false)));
                assertEquals(prefix.length + 64, state.position());
                MemoryView<MemorySegment> logits =
                        Views.castToSegmentBacked(model.logits(state), "logits");
                for (float value : Views.toFloatArray(logits, "logits"))
                    assertTrue(Float.isFinite(value));
            }
        }
    }

    private static Media.Image shiftedRainbow(int size) {
        float[] pixels = new float[size * size * 3];
        float cx = size / 2f, cy = size / 2f;
        float maxDistance = (float) Math.sqrt(cx * cx + cy * cy);
        for (int y = 0; y < size; y++)
            for (int x = 0; x < size; x++) {
                float dx = x - cx, dy = y - cy;
                float hue = (float) (Math.atan2(dy, dx) / (2f * 3.14159265f));
                if (hue < 0) hue += 1;
                float saturation = Math.min((float) Math.sqrt(dx * dx + dy * dy) / maxDistance, 1f);
                float h6 = hue * 6;
                int sector = (int) h6;
                float fraction = h6 - sector;
                float p = 1 - saturation;
                float q = 1 - saturation * fraction;
                float t = 1 - saturation * (1 - fraction);
                float[] rgb =
                        switch (sector % 6) {
                            case 0 -> new float[] {1, t, p};
                            case 1 -> new float[] {q, 1, p};
                            case 2 -> new float[] {p, 1, t};
                            case 3 -> new float[] {p, q, 1};
                            case 4 -> new float[] {t, p, 1};
                            default -> new float[] {1, p, q};
                        };
                for (int c = 0; c < 3; c++) pixels[(y * size + x) * 3 + c] = (rgb[c] + 1) * 0.5f;
            }
        return new Media.Image(pixels, size, size, 3);
    }
}
