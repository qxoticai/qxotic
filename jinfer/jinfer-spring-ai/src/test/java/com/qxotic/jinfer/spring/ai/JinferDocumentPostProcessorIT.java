package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.springframework.ai.content.Media;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.core.io.ByteArrayResource;
import org.springframework.util.MimeTypeUtils;

/**
 * The post-retrieval contract of {@link JinferDocumentPostProcessor} on Qwen3-Reranker 0.6B: what a
 * vector store handed over goes in, best-answers-first comes out. The corpus is the langchain4j
 * suite's - several chunks talk about returns, only one carries the window - so ordering is a
 * judgement about ANSWERING, not about topic. Model-gated via {@link ModelFixture}. Run: {@code mvn
 * test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-spring-ai}
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class JinferDocumentPostProcessorIT {

    private static final String ANSWER_FACT = "14 days";

    private static final Query QUESTION =
            new Query("I unpacked it already - how long before it is too late to send it back?");

    /** As a retriever would hand them over: similar-looking, one of them actually answering. */
    private static List<Document> retrieved() {
        return List.of(
                new Document(
                        "Return shipping: we email a prepaid return label once your return is"
                                + " registered. Drop the parcel at any pickup point; we do not"
                                + " collect returns from home addresses."),
                new Document(
                        "Refund policy: opened items can be returned within 14 days for store"
                                + " credit only. Unopened items are refunded in full within 30 days"
                                + " of purchase."),
                new Document(
                        "Gift returns: items bought as gifts can be exchanged by the recipient"
                                + " without a receipt, provided the original packaging is intact."),
                new Document(
                        "Shipping: standard delivery takes 3-5 business days. Express delivery"
                            + " arrives the next business day for orders placed before 15:00."));
    }

    static JinferDocumentPostProcessor processor;

    @BeforeAll
    static void load() {
        processor =
                JinferDocumentPostProcessor.builder()
                        .modelPath(ModelFixture.QWEN3_RERANKER_06B_Q8.require())
                        .contextLength(2048)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (processor != null) {
            processor.close();
            processor.close(); // idempotency pin: a second close must be a no-op, never ISE
        }
    }

    @Test
    void answeringDocumentComesFirstAndCarriesItsScore() {
        List<Document> ranked = processor.process(QUESTION, retrieved());
        assertEquals(4, ranked.size(), "nothing is dropped without a gate");
        assertTrue(ranked.get(0).getText().contains(ANSWER_FACT), ranked.get(0).getText());
        // scores are attached (overwriting any retrieval similarity) and sorted best first
        for (int i = 0; i < ranked.size(); i++) {
            Double score = ranked.get(i).getScore();
            assertTrue(score != null && score >= 0.0 && score <= 1.0, "score " + i + ": " + score);
            if (i > 0) {
                assertTrue(
                        ranked.get(i - 1).getScore() >= score, "descending: " + ranked.get(i - 1));
            }
        }
    }

    @Test
    void topKTruncatesAfterJudging() {
        // the point of reranking with topK: the kept document is not the one retrieval put first
        List<Document> ranked =
                JinferDocumentPostProcessor.builder()
                        .modelPath(ModelFixture.QWEN3_RERANKER_06B_Q8.require())
                        .contextLength(2048)
                        .topK(1)
                        .build()
                        .process(QUESTION, retrieved());
        assertEquals(1, ranked.size());
        assertTrue(ranked.get(0).getText().contains(ANSWER_FACT), ranked.get(0).getText());
    }

    @Test
    void minScoreEmptiesAnOffTopicContext() {
        JinferDocumentPostProcessor gated =
                JinferDocumentPostProcessor.builder()
                        .modelPath(ModelFixture.QWEN3_RERANKER_06B_Q8.require())
                        .contextLength(2048)
                        .minScore(0.5) // the verdict IS a probability: gate on it
                        .build();
        assertTrue(
                gated.process(new Query("Do you sell mountain bikes?"), retrieved()).isEmpty(),
                "no chunk answers an off-topic question - the gate must empty the context");
        assertTrue(gated.process(QUESTION, retrieved()).size() > 0, "a real answer survives it");
        gated.close();
    }

    @Test
    void nothingRetrievedPassesStraightThrough() {
        List<Document> empty = List.of();
        assertSame(empty, processor.process(QUESTION, empty), "no prefill, no copy");
    }

    @Test
    void mediaDocumentIsRefusedLoudly() {
        Document media =
                new Document(
                        new Media(
                                MimeTypeUtils.IMAGE_PNG,
                                new ByteArrayResource(new byte[] {1, 2, 3})),
                        Map.of());
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> processor.process(QUESTION, List.of(media)));
        assertTrue(e.getMessage().contains("media"), e.getMessage());
    }
}
