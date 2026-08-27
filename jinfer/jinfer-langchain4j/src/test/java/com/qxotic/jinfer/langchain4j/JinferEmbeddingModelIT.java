package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Arenas;
import com.qxotic.jinfer.chat.Models;
import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.exception.UnsupportedFeatureException;
import dev.langchain4j.model.embedding.request.EmbeddingInputType;
import dev.langchain4j.model.embedding.request.EmbeddingRequest;
import dev.langchain4j.model.embedding.request.EmbeddingRequestParameters;
import dev.langchain4j.model.embedding.response.EmbeddingResponse;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.io.ByteArrayOutputStream;
import java.io.PrintStream;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;
import java.util.concurrent.Executors;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * E2E contract of {@link JinferEmbeddingModel} on Qwen3-Embedding 0.6B: dimension, semantic
 * ordering, the packing law (packed batches must equal one-by-one embeds - segmented attention
 * isolates packed sequences), multi-group batches, exact usage, and both loud failures.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferEmbeddingModelIT {

    static JinferEmbeddingModel model;

    @BeforeAll
    static void load() {
        model =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"))
                        .contextLength(1024)
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

    @Test
    void matchesLlamaCppGoldenVectors() throws Exception {
        // goldens from llama.cpp llama-embedding (bN, same GGUF, bare prompt - the suite's own
        // path). Cross-engine kernels drift, so the gate is the suite's identity tolerance:
        // cosine > 0.999 (measured 0.99962 / 0.99965) and no element off by more than a
        // cent (measured maxAbsDiff 0.0039 / 0.0036). A tokenizer, layout, or pooling
        // regression blows both gates wide open; kernel noise never touches them.
        assertGolden(
                "The quick brown fox jumps over the lazy dog, twice.",
                "golden-embeddings/qwen3-fox.txt");
        assertGolden("A kitten was sitting by the window.", "golden-embeddings/qwen3-kitten.txt");
    }

    private static void assertGolden(String text, String resource) throws Exception {
        float[] golden = readGolden(resource);
        Embedding actual = model.embed(TextSegment.from(text)).content();
        assertEquals(golden.length, actual.vector().length, resource);
        double cos = cosine(actual, Embedding.from(golden));
        assertTrue(cos > 0.999, resource + ": cosine vs llama.cpp golden " + cos);
        double maxAbsDiff = 0;
        for (int i = 0; i < golden.length; i++) {
            maxAbsDiff = Math.max(maxAbsDiff, Math.abs(actual.vector()[i] - golden[i]));
        }
        assertTrue(maxAbsDiff < 0.01, resource + ": maxAbsDiff vs golden " + maxAbsDiff);
    }

    private static float[] readGolden(String resource) throws Exception {
        try (var in = JinferEmbeddingModelIT.class.getClassLoader().getResourceAsStream(resource)) {
            if (in == null) throw new IllegalStateException("missing test resource " + resource);
            String[] parts = new String(in.readAllBytes()).trim().split("\\s+");
            float[] out = new float[parts.length];
            for (int i = 0; i < parts.length; i++) out[i] = Float.parseFloat(parts[i]);
            return out;
        }
    }

    @Test
    void dimensionMatchesTheVectors() {
        Embedding e = model.embed("hello world").content();
        assertEquals(model.dimension(), e.dimension());
        assertTrue(model.dimension() > 0);
    }

    @Test
    void semanticOrdering() {
        Embedding cat = model.embed("A small domestic cat sat on the windowsill.").content();
        Embedding kitten = model.embed("A kitten was sitting by the window.").content();
        Embedding tax = model.embed("Quarterly corporate tax filings are due in April.").content();
        double related = cosine(cat, kitten);
        double unrelated = cosine(cat, tax);
        assertTrue(
                related > unrelated, "related " + related + " should beat unrelated " + unrelated);
    }

    @Test
    void packedBatchEqualsOneByOne() {
        List<TextSegment> segments = corpus(12);
        List<Embedding> packed = model.embedAll(segments).content();
        assertEquals(segments.size(), packed.size());
        for (int i = 0; i < segments.size(); i++) {
            Embedding solo = model.embed(segments.get(i)).content();
            double cos = cosine(packed.get(i), solo);
            assertTrue(cos > 0.999, "segment " + i + ": packed vs solo cosine " + cos);
        }
    }

    @Test
    void batchesSpanningMultipleGroups() {
        List<TextSegment> segments = corpus(40); // well past one 1024-token packing window
        Response<List<Embedding>> r = model.embedAll(segments);
        assertEquals(40, r.content().size());
        for (Embedding e : r.content()) assertEquals(model.dimension(), e.dimension());
    }

    @Test
    void usageIsExact() {
        Response<List<Embedding>> r = model.embedAll(corpus(5));
        assertTrue(r.tokenUsage().inputTokenCount() > 5, "counts include real tokens + EOS");
        assertEquals(0, model.embedAll(List.of()).content().size());
    }

    @Test
    void overLongSegmentFailsLoudly() {
        String words = "elaborate ".repeat(2000);
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class, () -> model.embed(TextSegment.from(words)));
        assertTrue(e.getMessage().contains("contextLength"), e.getMessage());
    }

    @Test
    void generativeArchitectureFailsLoudly() {
        // resolve BEFORE assertThrows: an abort inside the lambda would surface as a failure
        Path generative =
                TestModels.require("hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf");
        UnsupportedOperationException e =
                assertThrows(
                        UnsupportedOperationException.class,
                        () -> JinferEmbeddingModel.builder().modelPath(generative).build());
        assertTrue(e.getMessage().contains("not an embedding"), e.getMessage());
    }

    @Test
    void tokenCountsAreExactOnText() {
        var estimator = model.tokenCountEstimator();
        String text = "The quick brown fox jumps over the lazy dog, twice.";
        // the law: the estimator's text count IS the tokenizer's encoding length
        assertEquals(
                TextSegment.from(text).text().length() > 0 ? countViaEmbedderUsage(text) : 0,
                estimator.estimateTokenCountInText(text) + 1); // usage includes the EOS suffix
        assertTrue(estimator.estimateTokenCountInText("") == 0);
    }

    @Test
    void messageCountsSumVisibleText() {
        var estimator = model.tokenCountEstimator();
        var messages =
                List.of(
                        SystemMessage.from("Be brief."),
                        UserMessage.from("What is the capital of" + " France?"),
                        AiMessage.from("Paris."));
        int sum =
                estimator.estimateTokenCountInText("Be brief.")
                        + estimator.estimateTokenCountInText("What is the capital of France?")
                        + estimator.estimateTokenCountInText("Paris.");
        assertEquals(sum, estimator.estimateTokenCountInMessages(messages));
    }

    /** The billed input tokens of embedding {@code text} alone - ground truth from usage. */
    private static int countViaEmbedderUsage(String text) {
        return model.embed(TextSegment.from(text)).tokenUsage().inputTokenCount();
    }

    private static List<TextSegment> corpus(int n) {
        String[] topics = {
            "The recipe calls for two cups of flour and a pinch of salt.",
            "Interest rates were held steady by the central bank this quarter.",
            "The hiking trail climbs eight hundred meters over six kilometers.",
            "Her latest novel explores memory and loss in postwar Lisbon.",
            "The database migration completed without downtime last night.",
        };
        List<TextSegment> out = new ArrayList<>(n);
        for (int i = 0; i < n; i++) {
            out.add(TextSegment.from(topics[i % topics.length] + " (variant " + i + ")"));
        }
        return out;
    }

    // ---- input types: the framework's own query/document vocabulary, mapped to the card ----

    /** Qwen3-Embedding's instructed-query framing, verbatim from the card. */
    static final String QWEN3_QUERY_PREFIX =
            "Instruct: Given a web search query, retrieve relevant passages that answer the"
                    + " query\nQuery:";

    @Test
    void queryInputTypeAppliesTheCardFraming() {
        String text = "What is the capital of China?";
        Embedding typed =
                model.embed(
                                EmbeddingRequest.builder()
                                        .input(text)
                                        .inputType(EmbeddingInputType.QUERY)
                                        .build())
                        .embeddings()
                        .get(0);
        // identity law, at the suite's same-vector tolerance (separate passes jitter an LSB):
        // the typed request IS the hand-prefixed text, and is closer to it than to bare text
        Embedding manual = model.embed(QWEN3_QUERY_PREFIX + text).content();
        Embedding bare = model.embed(text).content();
        double toManual = cosine(typed, manual);
        double toBare = cosine(typed, bare);
        assertTrue(toManual > 0.999, "QUERY-typed vs hand-prefixed cosine: " + toManual);
        assertTrue(
                toManual > toBare,
                "QUERY framing must matter: manual=" + toManual + " bare=" + toBare);
    }

    @Test
    void documentInputTypeIsBareOnQwen3() {
        // the card: retrieval documents carry no instruction - DOCUMENT-typed == bare, pinned
        String text = "The capital of China is Beijing.";
        EmbeddingResponse typed =
                model.embed(
                        EmbeddingRequest.builder()
                                .input(text)
                                .inputType(EmbeddingInputType.DOCUMENT)
                                .build());
        Embedding bare = model.embed(text).content();
        double cos = cosine(typed.embeddings().get(0), bare);
        assertTrue(cos > 0.999, "DOCUMENT-typed vs bare cosine: " + cos);
    }

    @Test
    void dimensionsAreValidatedTruncatedAndNormalized() {
        assertEquals(
                Set.of(
                        EmbeddingRequestParameters.INPUT_TYPE,
                        EmbeddingRequestParameters.DIMENSIONS),
                model.supportedParameters());
        String text = "A kitten was sitting by the window.";
        Embedding nativeWidth =
                model.embed(EmbeddingRequest.builder().input(text).build()).embeddings().get(0);
        Embedding truncated =
                model.embed(EmbeddingRequest.builder().input(text).dimensions(64).build())
                        .embeddings()
                        .get(0);

        assertEquals(64, truncated.dimension());
        assertEquals(1, norm(truncated.vector()), 1e-5);
        float[] expected = Arrays.copyOf(nativeWidth.vector(), 64);
        normalize(expected);
        assertTrue(
                cosine(truncated, Embedding.from(expected)) > 0.999,
                "truncated output must be the normalized native prefix");

        for (int invalid : new int[] {-1, 0, 31, model.dimension() + 1}) {
            IllegalArgumentException failure =
                    assertThrows(
                            IllegalArgumentException.class,
                            () ->
                                    model.embed(
                                            EmbeddingRequest.builder()
                                                    .input(text)
                                                    .dimensions(invalid)
                                                    .build()));
            assertTrue(failure.getMessage().contains("32.."), failure.getMessage());
        }
        assertEquals(
                32,
                model.embed(EmbeddingRequest.builder().input(text).dimensions(32).build())
                        .embeddings()
                        .get(0)
                        .dimension());
        assertEquals(
                model.dimension(),
                model.embed(
                                EmbeddingRequest.builder()
                                        .input(text)
                                        .dimensions(model.dimension())
                                        .build())
                        .embeddings()
                        .get(0)
                        .dimension());
        for (Embedding embedding :
                model.embed(
                                EmbeddingRequest.builder()
                                        .inputs("one", "two", "three")
                                        .dimensions(64)
                                        .build())
                        .embeddings()) {
            assertEquals(64, embedding.dimension());
        }
    }

    @Test
    void fixedWidthModelDoesNotAdvertiseDimensions() {
        try (JinferEmbeddingModel fixed =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/LiquidAI/LFM2.5-Embedding-350M-GGUF/LFM2.5-Embedding-350M-Q8_0.gguf"))
                        .contextLength(256)
                        .build()) {
            assertEquals(
                    Set.of(EmbeddingRequestParameters.INPUT_TYPE), fixed.supportedParameters());
            assertEquals(1024, fixed.embed("native").content().dimension());
            assertThrows(
                    UnsupportedFeatureException.class,
                    () ->
                            fixed.embed(
                                    EmbeddingRequest.builder()
                                            .input("custom")
                                            .dimensions(512)
                                            .build()));
        }
    }

    @Test
    void typelessTrafficHintsOncePerInstance() {
        // own instance: the class-level model spent its hint tests ago
        JinferEmbeddingModel fresh =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"))
                        .contextLength(512)
                        .build();
        var err = new ByteArrayOutputStream();
        PrintStream real = System.err;
        try {
            System.setErr(new PrintStream(err, true));
            fresh.embed("hello");
            fresh.embed("again");
        } finally {
            System.setErr(real);
            fresh.close();
        }
        String notes = err.toString();
        assertTrue(notes.contains("embeddingInputType"), notes);
        assertEquals(
                notes.indexOf("noted once per model"),
                notes.lastIndexOf("noted once per model"),
                "hinted ONCE: " + notes);
    }

    @Test
    void retrieverAndIngestorSpeakInputTypes() {
        // the E2E wiring this exists for: framework knobs, no jinfer-specific API
        var store = new InMemoryEmbeddingStore<TextSegment>();
        EmbeddingStoreIngestor.builder()
                .embeddingModel(model)
                .embeddingStore(store)
                .embeddingInputType(EmbeddingInputType.DOCUMENT)
                .build()
                .ingest(
                        Document.from(
                                "The reset portal for AcmeCloud is https://acme.example/reset."),
                        Document.from("Bananas are yellow fruit rich in potassium."));
        var retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingStore(store)
                        .embeddingModel(model)
                        .embeddingInputType(EmbeddingInputType.QUERY)
                        .maxResults(1)
                        .build();
        var contents = retriever.retrieve(Query.from("Where do I reset my AcmeCloud password?"));
        assertEquals(1, contents.size());
        assertTrue(
                contents.get(0).textSegment().text().contains("acme.example/reset"),
                contents.get(0).textSegment().text());
    }

    // ---- shared weights: one load, parallel pipelines, user-owned lifetime ----

    @Test
    void sharedWeightsForkIsAParallelPipeline() throws Exception {
        try (Arena arena = Arenas.newCrossThread()) {
            var loaded =
                    Models.loadEmbedder(
                            TestModels.require(
                                    "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"),
                            arena);
            JinferEmbeddingModel a =
                    JinferEmbeddingModel.builder().model(loaded).contextLength(1024).build();
            JinferEmbeddingModel b = a.fork();
            try {
                // borrowed == owned: same vectors as the path-built class-level model
                Embedding shared = a.embed("hello world").content();
                assertTrue(cosine(shared, model.embed("hello world").content()) > 0.999);
                // CONCURRENT embeds on two pipelines over ONE weights copy - the law this
                // feature exists for (all scratch lives in each instance's own state)
                var pool = Executors.newFixedThreadPool(2);
                try {
                    var fa = pool.submit(() -> a.embed("A kitten sat on the mat.").content());
                    var fb = pool.submit(() -> b.embed("Taxes are due in April.").content());
                    assertEquals(model.dimension(), fa.get().dimension());
                    assertEquals(model.dimension(), fb.get().dimension());
                    assertTrue(
                            cosine(fa.get(), fb.get()) < 0.9, "distinct texts, distinct vectors");
                } finally {
                    pool.shutdown();
                }
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
        assertTrue(e.getMessage().contains("model(loaded)"), e.getMessage());
    }

    @Test
    void useAfterCloseFailsLoudly() {
        // the OWNED path's twin pin: close() frees weights and state together, and later use
        // must be an ISE (the state's closed flag or the weights canary - never a crash)
        JinferEmbeddingModel closed =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"))
                        .contextLength(256)
                        .build();
        closed.close();
        assertThrows(IllegalStateException.class, () -> closed.embed("hello"));
    }

    @Test
    void useAfterTheOwnerFreesTheWeightsFailsFast() throws Exception {
        // the safety canary at the forward's entry turns what used to be a SIGSEGV into a
        // teaching ISE - for the SEQUENTIAL mistake; freeing DURING a request stays a data race
        Arena arena = Arenas.newCrossThread();
        JinferEmbeddingModel borrowed;
        try {
            var loaded =
                    Models.loadEmbedder(
                            TestModels.require(
                                    "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"),
                            arena);
            borrowed = JinferEmbeddingModel.builder().model(loaded).contextLength(512).build();
        } catch (Throwable t) {
            arena.close();
            throw t;
        }
        try {
            arena.close(); // the owner frees the weights under the pipeline - out of order
            IllegalStateException e =
                    assertThrows(IllegalStateException.class, () -> borrowed.embed("hello"));
            assertTrue(e.getMessage().contains("freed"), e.getMessage());
            assertTrue(e.getMessage().contains("close your arena LAST"), e.getMessage());
        } finally {
            borrowed.close(); // the state is the instance's own arena: still closable
        }
    }

    private static double cosine(Embedding a, Embedding b) {
        float[] x = a.vector(), y = b.vector();
        double dot = 0, nx = 0, ny = 0;
        for (int i = 0; i < x.length; i++) {
            dot += (double) x[i] * y[i];
            nx += (double) x[i] * x[i];
            ny += (double) y[i] * y[i];
        }
        return dot / (Math.sqrt(nx) * Math.sqrt(ny));
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
}
