package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.testkit.TestModels;
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
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import com.qxotic.jinfer.chat.Models;
import java.lang.foreign.Arena;
import java.util.Arrays;
import java.util.Map;
import org.springframework.ai.content.Media;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeTypeUtils;

/**
 * E2E contract of {@link JinferEmbeddingModel} on Qwen3-Embedding 0.6B (mirrors the old suite):
 * dimension, semantic ordering, the packing law (packed batches must equal one-by-one embeds -
 * segmented attention isolates packed sequences), multi-group batches, exact usage, and both loud
 * failures. Model-gated via {@link TestModels}. Run: {@code mvn test -Dsurefire.excludedGroups=
 * -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferEmbeddingModelIT {

    private static final String EMBED_REF =
            "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf";

    static JinferEmbeddingModel model;
    static TestObservationRegistry observations;

    @BeforeAll
    static void load() {
        observations = TestObservationRegistry.create();
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(TestModels.require(EMBED_REF))
                        .contextLength(1024)
                        .observationRegistry(observations)
                        .build();
        // vectors are bit-stable only once the JIT settles (cold passes drift ~1 LSB); the
        // identity assertions below compare across calls, so warm up first
        model.embed("warmup");
        model.embed("warmup");
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

    private static double norm(float[] vector) {
        double squared = 0;
        for (float value : vector) squared += (double) value * value;
        return Math.sqrt(squared);
    }

    private static void normalize(float[] vector) {
        double scale = 1 / norm(vector);
        for (int i = 0; i < vector.length; i++) vector[i] *= (float) scale;
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
            // both sides through the raw door: embed(String) is the QUERY-framed face now
            float[] solo = raw(inputs.get(i));
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
    void dimensionsOptionIsValidatedTruncatedAndNormalized() {
        String text = "A kitten was sitting by the window.";
        float[] nativeWidth = raw(text);
        EmbeddingOptions options = EmbeddingOptions.builder().dimensions(64).build();
        float[] truncated =
                model.call(new EmbeddingRequest(List.of(text), options))
                        .getResults()
                        .get(0)
                        .getOutput();

        assertEquals(64, truncated.length);
        assertEquals(1, norm(truncated), 1e-5);
        float[] expected = Arrays.copyOf(nativeWidth, 64);
        normalize(expected);
        assertTrue(
                cosine(truncated, expected) > 0.999,
                "truncated output must be the normalized native prefix");

        for (int invalid : new int[] {-1, 0, 31, model.dimensions() + 1}) {
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    model.call(
                                            new EmbeddingRequest(
                                                    List.of(text),
                                                    EmbeddingOptions.builder()
                                                            .dimensions(invalid)
                                                            .build())));
            assertTrue(failure.getMessage().contains("32.."), failure.getMessage());
        }
        assertEquals(
                32,
                model.call(
                                new EmbeddingRequest(
                                        List.of(text),
                                        EmbeddingOptions.builder().dimensions(32).build()))
                        .getResults()
                        .get(0)
                        .getOutput()
                        .length);
        assertEquals(
                model.dimensions(),
                model.call(
                                new EmbeddingRequest(
                                        List.of(text),
                                        EmbeddingOptions.builder()
                                                .dimensions(model.dimensions())
                                                .build()))
                        .getResults()
                        .get(0)
                        .getOutput()
                        .length);
        for (Embedding embedding :
                model.call(new EmbeddingRequest(corpus(3), options)).getResults()) {
            assertEquals(64, embedding.getOutput().length);
        }
    }

    @Test
    void fixedWidthModelAcceptsOnlyItsNativeDimension() {
        try (JinferEmbeddingModel fixed =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf"))
                        .contextLength(256)
                        .build()) {
            assertEquals(
                    1024,
                    fixed.call(new EmbeddingRequest(List.of("native"), null))
                            .getResults()
                            .get(0)
                            .getOutput()
                            .length);
            assertEquals(
                    1024,
                    fixed.call(
                                    new EmbeddingRequest(
                                            List.of("explicit native"),
                                            EmbeddingOptions.builder().dimensions(1024).build()))
                            .getResults()
                            .get(0)
                            .getOutput()
                            .length);
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    fixed.call(
                                            new EmbeddingRequest(
                                                    List.of("custom"),
                                                    EmbeddingOptions.builder()
                                                            .dimensions(512)
                                                            .build())));
            assertTrue(failure.getMessage().contains("fixed embedding dimension"));
        }
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
                                        .modelPath(
                                                TestModels.require(
                                                        "hf.co/unsloth/gemma-4-E2B-it-GGUF:Q8_0"))
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

    // ---- the retrieval seam: the interface types state the intent (String = query side,
    // Document = ingestion side), mapped to the model card's framing ----

    /** Qwen3-Embedding's instructed-query framing, verbatim from the card. */
    static final String QWEN3_QUERY_PREFIX =
            "Instruct: Given a web search query, retrieve relevant passages that answer the"
                    + " query\nQuery:";

    @Test
    void typedOverloadsApplyTheCardFraming() {
        String text = "What is the capital of China?";
        // identity law, at the suite's same-vector tolerance (separate passes jitter an LSB):
        // the String overload IS the hand-prefixed query, and closer to it than to bare text
        float[] query = model.embed(text);
        double toManual = cosine(query, raw(QWEN3_QUERY_PREFIX + text));
        double toBare = cosine(query, raw(text));
        assertTrue(toManual > 0.999, "String overload vs hand-prefixed cosine: " + toManual);
        assertTrue(
                toManual > toBare,
                "query framing must matter: manual=" + toManual + " bare=" + toBare);
        // qwen3 documents are bare per the card: the Document overload == raw, pinned
        float[] document = model.embed(new Document(text));
        double doc = cosine(document, raw(text));
        assertTrue(doc > 0.999, "Document overload vs bare cosine: " + doc);
    }

    @Test
    void batchedDocumentEmbedMatchesSingle() {
        List<Document> docs =
                List.of(
                        new Document("The cat sat on the warm windowsill."),
                        new Document("Quarterly tax filings are due in April."));
        // one batch (List::of); the store route must equal the single-Document route
        List<float[]> batched = model.embed(docs, EmbeddingOptions.builder().build(), List::of);
        assertEquals(2, batched.size());
        for (int i = 0; i < docs.size(); i++) {
            assertTrue(cosine(batched.get(i), model.embed(docs.get(i))) > 0.999);
        }
    }

    @Test
    void vectorStoreIsRetrievalCorrectBothSides() {
        // the single-bean wiring Spring forces: ONE model, both sides framed correctly anyway
        var store = SimpleVectorStore.builder(model).build();
        store.add(
                List.of(
                        new Document(
                                "The reset portal for AcmeCloud is https://acme.example/reset."),
                        new Document("Bananas are yellow fruit rich in potassium.")));
        List<Document> hits =
                store.similaritySearch(
                        SearchRequest.builder()
                                .query("Where do I reset my AcmeCloud password?")
                                .topK(1)
                                .build());
        assertEquals(1, hits.size());
        assertTrue(hits.get(0).getText().contains("acme.example/reset"), hits.get(0).getText());
    }

    @Test
    void sharedWeightsForkIsAParallelPipeline() throws Exception {
        // ONE load in the USER's arena; fork() mints a second pipeline for a context's price
        try (Arena arena = Arenas.newCrossThread()) {
            var loaded =
                    Models.loadEmbedder(
                            TestModels.require(EMBED_REF), arena);
            JinferEmbeddingModel a =
                    JinferEmbeddingModel.builder().model(loaded).contextLength(1024).build();
            JinferEmbeddingModel b = a.fork();
            try {
                float[] shared = a.embed(new Document("hello world"));
                assertTrue(cosine(shared, model.embed(new Document("hello world"))) > 0.999);
                assertEquals(model.dimensions(), b.embed(new Document("second pipeline")).length);
            } finally {
                a.close();
                b.close();
            }
        }
    }

    @Test
    void forkOfAnOwningModelRefusesWithTheRecipe() {
        IllegalStateException e = assertThrows(IllegalStateException.class, model::fork);
        assertTrue(e.getMessage().contains("Models.loadEmbedder"), e.getMessage());
    }

    @Test
    void mediaDocumentIsRefusedLoudly() {
        Document media =
                new Document(
                        new Media(
                                MimeTypeUtils.IMAGE_PNG,
                                new ByteArrayResource(
                                        new byte[] {1, 2, 3})),
                        Map.of());
        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> model.embed(media));
        assertTrue(e.getMessage().contains("media"), e.getMessage());
        IllegalArgumentException batched =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                model.embed(
                                        List.of(media),
                                        EmbeddingOptions.builder().build(),
                                        List::of));
        assertTrue(batched.getMessage().contains("media"), batched.getMessage());
    }

    @Test
    void embedForResponseStaysRaw() {
        // the seam contract, pinned from the other side: only embed(String) is query-framed;
        // embedForResponse rides call(), the raw framing-free door
        String text = "What is the capital of China?";
        float[] viaResponse = model.embedForResponse(List.of(text)).getResults().get(0).getOutput();
        assertTrue(cosine(viaResponse, raw(text)) > 0.999, "embedForResponse must stay raw");
        double toQuery = cosine(viaResponse, model.embed(text));
        double toRaw = cosine(viaResponse, raw(text));
        assertTrue(toRaw > toQuery, "raw=" + toRaw + " query=" + toQuery);
    }

    private static float[] raw(String text) {
        return model.call(new EmbeddingRequest(List.of(text), null))
                .getResults()
                .get(0)
                .getOutput();
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
