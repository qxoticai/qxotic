package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.LoadedEmbedder;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.embedding.EmbeddingModel;
import java.lang.foreign.Arena;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The builder's {@code dimensions(int)} knob, which an embedding store relies on: a call that names
 * no width embeds at the builder's, {@link JinferEmbeddingModel#dimension()} reports it, and a
 * request that names its own still wins. EmbeddingGemma 2 is Matryoshka-trained, 128..768.
 */
@Tag("integration")
final class JinferEmbeddingModelDimensionsIT {

    static final String MODEL = "hf.co/unsloth/embeddinggemma-2-GGUF/embeddinggemma-2-Q8_0.gguf";
    static final int NATIVE = 768, MINIMUM = 128;

    static final Arena ARENA = Arena.ofShared();
    static LoadedEmbedder<?> weights;

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
            assertEquals(256, model.dimension(), "reported, so a store sizes its table from it");

            // the default path a store uses: no per-call width
            assertEquals(256, model.embed("hello").content().dimension());
            // the batched door embeds at the same width
            assertEquals(
                    256,
                    model.embedAll(List.of(TextSegment.from("a"), TextSegment.from("b")))
                            .content()
                            .getFirst()
                            .dimension());
        }
    }

    @Test
    void theModelsOwnWidthIsTheDefault() {
        try (JinferEmbeddingModel model = JinferEmbeddingModel.builder().model(weights).build()) {
            assertEquals(NATIVE, model.dimension());
            assertEquals(NATIVE, model.embed("hello").content().dimension());
        }
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(0).build()) {
            assertEquals(NATIVE, model.dimension(), "0 is the model's own width");
        }
    }

    /** A Matryoshka scale of widths through the builder: each call is that long. */
    @Test
    void everyLegalWidthEmbedsAtThatWidth() {
        for (int width : new int[] {MINIMUM, 256, 512, NATIVE}) {
            try (JinferEmbeddingModel model =
                    JinferEmbeddingModel.builder().model(weights).dimensions(width).build()) {
                float[] vector = model.embed("a sentence to embed").content().vector();
                assertEquals(width, vector.length, "width " + width);
                for (float v : vector) assertTrue(Float.isFinite(v), "width " + width);
            }
        }
    }

    /** The narrow vector is the prefix of the wide one: Matryoshka truncation, not a re-embed. */
    @Test
    void aNarrowWidthIsThePrefixOfTheWideOne() {
        float[] full;
        float[] narrow;
        try (JinferEmbeddingModel model = JinferEmbeddingModel.builder().model(weights).build()) {
            full = model.embed("a sentence to embed").content().vector();
        }
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(MINIMUM).build()) {
            narrow = model.embed("a sentence to embed").content().vector();
        }
        double dot = 0, a = 0, b = 0;
        for (int i = 0; i < MINIMUM; i++) {
            dot += full[i] * narrow[i];
            a += full[i] * full[i];
            b += narrow[i] * narrow[i];
        }
        assertEquals(1.0, dot / Math.sqrt(a * b), 1e-4, "the narrow vector is the prefix");
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

    /** The response carries the width embedded, which a store reads back per vector. */
    @Test
    void theResponseCarriesTheWidth() {
        try (JinferEmbeddingModel model =
                JinferEmbeddingModel.builder().model(weights).dimensions(256).build()) {
            Embedding embedding = model.embed("hello").content();
            assertEquals(256, embedding.dimension());
            assertTrue(model instanceof EmbeddingModel, "the framework's own interface");
        }
    }
}
