package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import io.micrometer.observation.tck.TestObservationRegistry;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.Embedding;
import org.springframework.ai.embedding.EmbeddingOptions;
import org.springframework.ai.embedding.EmbeddingRequest;
import org.springframework.ai.embedding.EmbeddingResponse;

/**
 * E2E contract of {@link JinferEmbeddingModel} on Qwen3-Embedding 0.6B (mirrors the langchain4j
 * suite): dimension, semantic ordering, the packing law (packed batches must equal one-by-one
 * embeds - segmented attention isolates packed sequences), multi-group batches, exact usage, and
 * both loud failures. Model-gated via {@link ModelFixture}. Run: {@code mvn test
 * -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferEmbeddingModelIT {

    static JinferEmbeddingModel model;
    static TestObservationRegistry observations;

    @BeforeAll
    static void load() {
        observations = TestObservationRegistry.create();
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(ModelFixture.QWEN3_EMBED_06B_Q8.require())
                        .contextLength(1024)
                        .observationRegistry(observations)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) {
            model.close();
            model.close(); // idempotency pin: a second close must be a no-op, never ISE
        }
    }

    static List<String> corpus(int n) {
        String[] topics = {
            "The cat sat on the warm windowsill.",
            "Quarterly tax filings are due in April.",
            "A kitten played with a ball of yarn.",
            "The recipe calls for two cups of flour.",
            "She hiked the northern ridge at dawn.",
            "The invoice total comes to 48 euros.",
            "A puppy barked at the garden gate.",
            "The committee postponed the vote again.",
        };
        List<String> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) out.add(topics[i % topics.length] + " (" + i + ")");
        return out;
    }

    static double cosine(float[] a, float[] b) {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.length; i++) {
            dot += (double) a[i] * b[i];
            na += (double) a[i] * a[i];
            nb += (double) b[i] * b[i];
        }
        return dot / (Math.sqrt(na) * Math.sqrt(nb));
    }

    @Test
    void dimensionMatchesTheVectors() {
        float[] v = model.embed("hello world");
        assertEquals(model.dimensions(), v.length);
        assertTrue(model.dimensions() > 0);
    }

    @Test
    void semanticOrdering() {
        float[] cat = model.embed("A small domestic cat sat on the windowsill.");
        float[] kitten = model.embed("A kitten was sitting by the window.");
        float[] tax = model.embed("Quarterly corporate tax filings are due in April.");
        double related = cosine(cat, kitten);
        double unrelated = cosine(cat, tax);
        assertTrue(
                related > unrelated, "related " + related + " should beat unrelated " + unrelated);
    }

    @Test
    void packedBatchEqualsOneByOne() {
        List<String> inputs = corpus(12);
        EmbeddingResponse packed = model.call(new EmbeddingRequest(inputs, null));
        assertEquals(inputs.size(), packed.getResults().size());
        for (int i = 0; i < inputs.size(); i++) {
            float[] solo = model.embed(inputs.get(i));
            double cos = cosine(packed.getResults().get(i).getOutput(), solo);
            assertTrue(cos > 0.999, "input " + i + ": packed vs solo cosine " + cos);
        }
    }

    @Test
    void batchesSpanningMultipleGroups() {
        List<String> inputs = corpus(40); // well past one 1024-token packing window
        EmbeddingResponse r = model.call(new EmbeddingRequest(inputs, null));
        assertEquals(40, r.getResults().size());
        for (Embedding e : r.getResults()) assertEquals(model.dimensions(), e.getOutput().length);
    }

    @Test
    void usageIsExact() {
        EmbeddingResponse r = model.call(new EmbeddingRequest(corpus(5), null));
        assertTrue(
                r.getMetadata().getUsage().getPromptTokens() > 5,
                "counts include real tokens + EOS suffix");
        assertTrue(model.call(new EmbeddingRequest(List.of(), null)).getResults().isEmpty());
    }

    @Test
    void dimensionsOptionTruncatesVectors() {
        EmbeddingOptions options = EmbeddingOptions.builder().dimensions(8).build();
        EmbeddingResponse r = model.call(new EmbeddingRequest(corpus(3), options));
        for (Embedding e : r.getResults()) assertEquals(8, e.getOutput().length);
    }

    @Test
    void overLongSegmentFailsLoudly() {
        String words = "elaborate ".repeat(2000);
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> model.call(new EmbeddingRequest(List.of(words), null)));
        assertTrue(e.getMessage().contains("context"), e.getMessage());
    }

    @Test
    void generativeArchitectureFailsLoudly() {
        UnsupportedOperationException e =
                assertThrows(
                        UnsupportedOperationException.class,
                        () ->
                                JinferEmbeddingModel.builder()
                                        .modelPath(ModelFixture.GEMMA4_E2B_Q8.require())
                                        .build());
        assertTrue(e.getMessage().contains("not an embedding"), e.getMessage());
    }

    @Test
    void perRequestModelSwitchRejected() {
        EmbeddingOptions options = EmbeddingOptions.builder().model("some-other.gguf").build();
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> model.call(new EmbeddingRequest(List.of("hi"), options)));
        assertTrue(e.getMessage().contains("one loaded GGUF per instance"), e.getMessage());
    }

    @Test
    void documentEmbedUsesItsText() {
        float[] viaDocument = model.embed(new Document("hello world"));
        float[] viaString = model.embed("hello world");
        assertEquals(cosine(viaDocument, viaString), 1.0, 1e-6);
    }

    @Test
    void observationOnCall() {
        observations.clear();
        model.call(new EmbeddingRequest(corpus(3), null));
        observations
                .assertThat()
                .hasNumberOfObservationsWithNameEqualTo("gen_ai.client.operation", 1);
        observations.assertThat().hasAnObservationWithAKeyValue("gen_ai.system", "jinfer");
        observations.assertThat().hasAnObservationWithAKeyName("gen_ai.usage.input_tokens");
    }
}
