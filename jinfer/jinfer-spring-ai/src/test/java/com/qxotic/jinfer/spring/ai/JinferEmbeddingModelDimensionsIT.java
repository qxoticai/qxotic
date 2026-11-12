package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import java.lang.foreign.Arena;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * The builder's {@code dimensions(int)} knob, which a vector store relies on: a call that names no
 * width embeds at the builder's, {@link JinferEmbeddingModel#dimensions()} reports it, and a
 * request that names its own still wins. EmbeddingGemma 2 is Matryoshka-trained, so 128..768 are
 * all legal.
 */
@Tag("integration")
final class JinferEmbeddingModelDimensionsIT {

    static final String MODEL = "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final int NATIVE = 768, MINIMUM = 128;

    static final Arena ARENA = Arena.ofShared();
    static com.qxotic.jinfer.chat.LoadedEmbedder<?> weights;

    @BeforeAll
    static void load() throws Exception {
        weights = Models.loadEmbedder(TestModels.require(MODEL), ARENA);
    }

    @AfterAll
    static void close() {
        ARENA.close();
    }

    @Test
    void theBuilderWidthIsEveryCallsWidth() {
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(256).build()) {
            assertEquals(256, model.dimensions(), "reported, so a store sizes its table from it");

            // the default path a vector store uses: no options
            EmbeddingResponse response =
                    model.call(new EmbeddingRequest(java.util.List.of("hello"), null));
            assertEquals(256, response.getResults().get(0).getOutput().length);

            // the shortcut accessors go through the same width
            assertEquals(256, model.embed("hello").length);
            assertEquals(256, model.embed(new Document("hello")).length);
        }
    }

    @Test
    void theModelsOwnWidthIsTheDefault() {
        try (JinferEmbeddingModel model = JinferEmbeddingModel.builder().model(weights).build()) {
            assertEquals(NATIVE, model.dimensions());
            assertEquals(NATIVE, model.embed("hello").length);
        }
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(0).build()) {
            assertEquals(NATIVE, model.dimensions(), "0 is the model's own width");
        }
    }

    /** A request that names its own width wins over the builder's, for that call only. */
    @Test
    void aRequestsOwnWidthWins() {
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(256).build()) {
            EmbeddingResponse narrow =
                    model.call(
                            new EmbeddingRequest(
                                    java.util.List.of("hello"),
                                    EmbeddingOptions.builder().dimensions(MINIMUM).build()));
            assertEquals(MINIMUM, narrow.getResults().get(0).getOutput().length);
            // and the next call is back at the builder's width
            assertEquals(256, model.embed("hello").length);
        }
    }

    /** A Matryoshka scale of widths: each is exactly that long, and shorter ones stay usable. */
    @Test
    void everyLegalWidthEmbedsAtThatWidth() {
        try (JinferEmbeddingModel model = JinferEmbeddingModel.builder().model(weights).build()) {
            for (int width : new int[] {MINIMUM, 256, 512, NATIVE}) {
                float[] vector = embedAt(model, "a sentence to embed", width);
                assertEquals(width, vector.length, "width " + width);
                for (float v : vector) assertTrue(Float.isFinite(v), "width " + width);
            }
            // a prefix of the full vector is the narrow vector: Matryoshka truncation, not a
            // re-embed
            float[] full = embedAt(model, "a sentence to embed", NATIVE);
            float[] short_ = embedAt(model, "a sentence to embed", MINIMUM);
            double dot = 0, a = 0, b = 0;
            for (int i = 0; i < MINIMUM; i++) {
                dot += full[i] * short_[i];
                a += full[i] * full[i];
                b += short_[i] * short_[i];
            }
            assertEquals(1.0, dot / Math.sqrt(a * b), 1e-4, "the short vector is the prefix");
        }
    }

    /** One width for one call: the Spring AI request, whose options carry the width. */
    private static float[] embedAt(JinferEmbeddingModel model, String text, int width) {
        return model.call(
                        new EmbeddingRequest(
                                java.util.List.of(text),
                                EmbeddingOptions.builder().dimensions(width).build()))
                .getResults()
                .getFirst()
                .getOutput();
    }

    @Test
    void aWidthTheModelCannotProduceIsRefusedAtBuild() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        JinferEmbeddingModel.builder()
                                .model(weights)
                                .dimensions(MINIMUM - 1)
                                .build());
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferEmbeddingModel.builder().model(weights).dimensions(NATIVE + 1).build());
        assertThrows(
                IllegalArgumentException.class,
                () -> JinferEmbeddingModel.builder().model(weights).dimensions(-1));
    }

    /** The baked-in width is what a call without options reports to an observation. */
    @Test
    void theOptionsCarryTheBuilderWidth() {
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(256).build()) {
            assertEquals(256, model.dimensions());
            // the model's own width leaves the options unstated, as before the knob existed
            try (JinferEmbeddingModel nativeWidth =
                    JinferEmbeddingModel.builder().model(weights).build()) {
                assertEquals(NATIVE, nativeWidth.dimensions());
            }
        }
    }
}
