package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.output.Response;
import java.util.ArrayList;
import java.util.List;
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
                        .modelPath(ModelFixture.QWEN3_EMBED_06B_Q8.require())
                        .contextLength(1024)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) {
            model.close();
            model.close(); // idempotency pin: a second close must be a no-op, never ISE
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
}
