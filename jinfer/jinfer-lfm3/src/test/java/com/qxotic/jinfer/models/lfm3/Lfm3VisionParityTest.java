package com.qxotic.jinfer.models.lfm3;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.codecs.ImageCodec;
import com.qxotic.jinfer.kernels.Resample;
import com.qxotic.jinfer.media.Media;
import com.qxotic.jinfer.testkit.TestModels;
import com.qxotic.jota.memory.MemoryArena;
import com.qxotic.jota.memory.MemoryView;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

/**
 * The LFM3-VL tower against llama.cpp's mtmd on the same F16 mmproj: the resized, normalized pixels
 * exactly, then every block's residual and the projected rows.
 */
@Tag("integration")
final class Lfm3VisionParityTest {

    static final String TEXT = "hf.co/LiquidAI/LFM3-VL-2B-GGUF/LFM3-VL-2B-Q8_0.gguf";
    static final String MMPROJ = "hf.co/LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-F16.gguf";
    static final String MMPROJ_F32 = "hf.co/LiquidAI/LFM3-VL-2B-GGUF/mmproj-LFM3-VL-2B-F32.gguf";

    /**
     * F32 weights on both sides: what is left is the order of float sums and the F16 K/V of ggml's
     * flash attention, which jinfer reads in F16 too.
     */
    @ParameterizedTest
    @ValueSource(strings = {"cat", "small"})
    void theF32TowerMatchesLlamaCppToTheArithmetic(String name) throws Exception {
        check("vl32-" + name, name, MMPROJ_F32, 1e-3, 0.999999);
    }

    /**
     * The shipped F16 mmproj: ggml rounds every matmul's activations to F16 too, jinfer keeps them
     * in F32, so the bound is that rounding's, compounded over 14 blocks.
     */
    @ParameterizedTest
    @ValueSource(strings = {"cat", "dice", "small", "wide"})
    void theF16TowerTracksLlamaCpp(String name) throws Exception {
        check("vl-" + name, name, MMPROJ, 2e-2, 0.9998);
    }

    private static void check(
            String dump, String name, String mmproj, double layerBound, double cosBound)
            throws Exception {
        Path dir = Oracle.dir(dump);
        Media.Image image = ImageCodec.load(TestModels.fixture("lfm3/images/" + name + ".png"));
        try (Arena arena = Arena.ofShared()) {
            Lfm3 model =
                    Lfm3.loadModel(TestModels.require(TEXT), TestModels.require(mmproj), arena);
            Lfm3Vision vision = (Lfm3Vision) model.projector(Media.Image.class).orElseThrow();
            Oracle.Tensor pixels = Oracle.tensor(dir, "inp_raw");
            int width = Math.toIntExact(pixels.ne()[0]), height = Math.toIntExact(pixels.ne()[1]);
            int[] rgb = Resample.bicubic(image, width, height, true);
            float[] planar = new float[pixels.data().length];
            for (int p = 0; p < width * height; p++)
                for (int c = 0; c < 3; c++)
                    planar[c * width * height + p] = (rgb[p * 3 + c] / 255f - 0.5f) / 0.5f;
            Oracle.Score pixelScore = Oracle.score(pixels.data(), planar, width);
            System.out.printf("%s %dx%d pixels %s%n", name, width, height, pixelScore);
            assertEquals(0.0, pixelScore.maxRel(), "resized pixels");

            float[] expected = Oracle.floats(dir.resolve("embd.f32"));
            int dim = model.configuration().embeddingLength();
            assertEquals(expected.length / dim, vision.positions(image));
            MemoryArena<MemorySegment> scratch = Arenas.newCrossThreadMemoryArena();
            try {
                double[] worst = {0};
                MemoryView<MemorySegment> rows =
                        vision.encode(
                                image,
                                scratch,
                                (x, layer) -> {
                                    try {
                                        Oracle.Tensor ref =
                                                Oracle.tensor(dir, "layer_out-" + layer);
                                        Oracle.Score s =
                                                Oracle.score(
                                                        ref.data(),
                                                        Oracle.rows(x, ref.rows(), ref.width()),
                                                        ref.width());
                                        System.out.printf("  layer_out-%-2d %s%n", layer, s);
                                        worst[0] = Math.max(worst[0], s.maxRel());
                                    } catch (IOException e) {
                                        throw new UncheckedIOException(e);
                                    }
                                });
                Oracle.Score out =
                        Oracle.score(expected, Oracle.rows(rows, expected.length / dim, dim), dim);
                System.out.printf("  embd %s%n", out);
                assertTrue(worst[0] < layerBound, "layer drift " + worst[0]);
                assertTrue(out.minCos() > cosBound, "rows " + out);
            } finally {
                Arenas.close(scratch);
            }
        }
    }
}
