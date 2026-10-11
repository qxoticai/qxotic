package com.qxotic.jinfer.models.gemmaembedding2;

import static com.qxotic.jinfer.models.gemmaembedding2.GemmaEmbedding2ParityTest.cosine;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertInstanceOf;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.codecs.ImageCodec;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
final class GemmaEmbedding2IntegrationTest {
    private static final Path IMAGES =
            Path.of("../jinfer-langchain4j/src/test/resources/kit-images");

    @Test
    void embedderPacksTextLikeOneAtATimeAndRetrieves() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> embedder =
                    Models.loadEmbedder(
                            GemmaEmbedding2ParityTest.path(
                                    "model", GemmaEmbedding2ParityTest.MODEL_REF),
                            arena);
            GemmaEmbedding2 model = assertInstanceOf(GemmaEmbedding2.class, embedder.model());
            assertEquals(768, embedder.dimension());
            assertEquals(128, embedder.minimumDimension());

            List<String> texts =
                    List.of(
                            embedder.queryPrefix() + "What causes the northern lights?",
                            embedder.documentPrefix()
                                    + "Auroras appear when charged particles from the sun hit the"
                                    + " upper atmosphere.",
                            embedder.documentPrefix()
                                    + "A good sourdough needs a lively starter and a long proof.",
                            // past the 1024-wide window and across query blocks of the pack
                            embedder.documentPrefix() + "Auroras and solar wind. ".repeat(400));
            List<float[]> packed = new ArrayList<>();
            try (GemmaEmbedding2.State state = model.newState(4_096, 512)) {
                embedder.embedAll(state, texts, packed::add);
                for (int i = 0; i < texts.size(); i++) {
                    float[] alone = embed(model, state, texts.get(i));
                    // the same sequence, packed or alone: bit-identical on F32 weights, but
                    // jam-native's int8 activation requant depends on the batch shape
                    double same = cosine(alone, packed.get(i));
                    System.out.printf("text %d packed~alone %.9f%n", i, same);
                    assertTrue(same > 0.9995, "text " + i + " packed~alone " + same);
                }
            }
            double aurora = cosine(packed.get(0), packed.get(1));
            double bread = cosine(packed.get(0), packed.get(2));
            System.out.printf("query~aurora %.4f, query~sourdough %.4f%n", aurora, bread);
            assertTrue(aurora > bread + 0.1, "retrieval ranks the answering document first");
        }
    }

    @Test
    void imagesLandNextToTheirCaptions() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            GemmaEmbedding2 model =
                    GemmaEmbedding2.loadModel(
                                    GemmaEmbedding2ParityTest.path(
                                            "model", GemmaEmbedding2ParityTest.MODEL_REF),
                                    arena)
                            .withMedia(
                                    GemmaEmbedding2ParityTest.path(
                                            "mmproj", GemmaEmbedding2ParityTest.MMPROJ_REF),
                                    arena);
            try (GemmaEmbedding2.State state = model.newState(1_024, 512)) {
                float[] cat = embed(model, state, ImageCodec.load(IMAGES.resolve("cat.png")));
                float[] dice = embed(model, state, ImageCodec.load(IMAGES.resolve("dice.png")));
                float[] catCaption = embed(model, state, "task: search result | query: a cat");
                float[] diceCaption = embed(model, state, "task: search result | query: dice");
                System.out.printf(
                        "cat~'a cat' %.4f cat~'dice' %.4f dice~'dice' %.4f dice~'a cat' %.4f%n",
                        cosine(cat, catCaption),
                        cosine(cat, diceCaption),
                        cosine(dice, diceCaption),
                        cosine(dice, catCaption));
                assertTrue(cosine(cat, catCaption) > cosine(cat, diceCaption));
                assertTrue(cosine(dice, diceCaption) > cosine(dice, catCaption));
            }
        }
    }

    @Test
    void aMediaCompanionLetsTheEmbedderTakeParts() throws Exception {
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> embedder =
                    Models.loadEmbedder(
                            GemmaEmbedding2ParityTest.path(
                                    "model", GemmaEmbedding2ParityTest.MODEL_REF),
                            arena,
                            Map.of(
                                    "media",
                                    GemmaEmbedding2ParityTest.path(
                                            "mmproj", GemmaEmbedding2ParityTest.MMPROJ_REF)),
                            null);
            GemmaEmbedding2 model = assertInstanceOf(GemmaEmbedding2.class, embedder.model());
            List<Object> parts =
                    List.of("A photo of a cat: ", ImageCodec.load(IMAGES.resolve("cat.png")));
            try (GemmaEmbedding2.State state = model.newState(1_024, 512)) {
                // the first pass runs the kernels interpreted, an ulp off the compiled ones
                GemmaEmbedding2ParityTest.embed(model, state, parts);
                float[] direct = GemmaEmbedding2ParityTest.embed(model, state, parts);
                List<float[]> full = new ArrayList<>(), shortened = new ArrayList<>();
                int positions = embedder.embed(state, parts, 768, full::add);
                embedder.embed(state, parts, 256, shortened::add);
                int caption = embedder.tokenizer().encode("A photo of a cat: ").length();
                assertEquals(1 + caption + 2 + 255 + 1, positions, "bos, text, image block, eos");
                assertArrayEquals(direct, full.getFirst(), 1e-6f, "the port's own embedding");
                float[] prefix = Arrays.copyOf(direct, 256);
                double norm = 0;
                for (float v : prefix) norm += v * v;
                for (int i = 0; i < prefix.length; i++) prefix[i] /= (float) Math.sqrt(norm);
                assertArrayEquals(prefix, shortened.getFirst(), 1e-6f, "Matryoshka, renormalized");
            }
        }
        try (Arena arena = Arena.ofShared()) {
            LoadedEmbedder<?> textOnly =
                    Models.loadEmbedder(
                            GemmaEmbedding2ParityTest.path(
                                    "model", GemmaEmbedding2ParityTest.MODEL_REF),
                            arena);
            try (var state = textOnly.model().newState(256)) {
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                textOnly.embed(
                                        state,
                                        List.of(ImageCodec.load(IMAGES.resolve("cat.png"))),
                                        768,
                                        v -> {}));
            }
        }
    }

    private static float[] embed(GemmaEmbedding2 model, GemmaEmbedding2.State state, Object part) {
        return GemmaEmbedding2ParityTest.embed(model, state, List.of(part));
    }
}
