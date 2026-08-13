package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.AugmentationRequest;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.aggregator.ReRankingContentAggregator;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Metadata;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * Retrieve-then-rerank, the standard two-stage RAG shape, fully local: {@link JinferEmbeddingModel}
 * casts a wide cheap net over the corpus, {@link JinferScoringModel} judges each candidate against
 * the question, and the chat model answers from what survived - three GGUFs in one JVM, no network.
 *
 * <p>The corpus is built so retrieval alone is NOT enough: several chunks talk about returns, but
 * only one carries the 14-day window and the store-credit outcome. Embedding similarity ranks by
 * topic; the reranker reads the pair and ranks by whether the chunk ANSWERS the question - which is
 * exactly the gap a reranker exists to close.
 *
 * <p>Graded, simplest first:
 *
 * <ol>
 *   <li>Rerank a retrieved shortlist by hand ({@code scoreAll}).
 *   <li>The same thing through langchain4j's {@link ReRankingContentAggregator}.
 *   <li>{@code minScore} as a relevance gate: an off-topic question retrieves chunks and keeps
 *       none.
 *   <li>The full stack: AiServices answering from the reranked contents.
 * </ol>
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class RerankRetrievalIT {

    /** Only this chunk carries the window; the others are topical near-misses. */
    private static final String ANSWER_FACT = "14 days";

    /**
     * Deliberately phrased AWAY from the answering chunk's wording ("unpacked"/"send back" vs
     * "opened"/"returned", no mention of days): embedding similarity puts the return-SHIPPING chunk
     * first, which cannot answer it. Measured on this corpus with Qwen3-Embedding-0.6B: the
     * answering chunk lands at embedding rank 1, and the reranker scores it 0.79 against 0.003 for
     * the chunk retrieval preferred.
     */
    private static final String QUESTION =
            "I unpacked it already - how long before it is too late to send it back?";

    static JinferEmbeddingModel embedder;
    static JinferScoringModel reranker;
    static JinferChatModel chat;
    static InMemoryEmbeddingStore<TextSegment> store;

    interface Assistant {
        String answer(String question);
    }

    @BeforeAll
    static void buildStack() {
        embedder =
                JinferEmbeddingModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF/Qwen3-Embedding-0.6B-Q8_0.gguf"))
                        .build();
        reranker =
                JinferScoringModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/mradermacher/Qwen3-Reranker-0.6B-GGUF/Qwen3-Reranker-0.6B.Q8_0.gguf"))
                        .contextLength(2048)
                        .build();
        chat =
                JinferChatModel.builder()
                        .modelPath(
                                TestModels.require(
                                        "hf.co/unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q8_0.gguf"))
                        .contextLength(4096)
                        .temperature(0.0)
                        .maxOutputTokens(160)
                        .build();
        store = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor.builder()
                // one FAQ entry = one chunk (every entry is well under the window): splitting a
                // policy mid-way would leave the two halves of one answer competing with each
                // other, which is a chunking problem, not a ranking one
                .documentSplitter(DocumentSplitters.recursive(400, 0))
                .embeddingModel(embedder)
                .embeddingStore(store)
                .build()
                .ingest(
                        List.of(
                                // the one that answers
                                Document.from(
                                        "Refund policy: opened items can be returned within 14 days"
                                            + " for store credit only. Unopened items are refunded"
                                            + " in full within 30 days of purchase."),
                                // near-miss: all about returns, answers nothing about the window
                                Document.from(
                                        "Return shipping: we email a prepaid return label once your"
                                            + " return is registered. Drop the parcel at any pickup"
                                            + " point; we do not collect returns from home"
                                            + " addresses."),
                                // near-miss: returns again, different subject
                                Document.from(
                                        "Gift returns: items bought as gifts can be exchanged by"
                                                + " the recipient without a receipt, provided the"
                                                + " original packaging is intact and undamaged."),
                                Document.from(
                                        "Shipping: standard delivery takes 3-5 business days."
                                            + " Express delivery arrives the next business day for"
                                            + " orders placed before 15:00."),
                                Document.from(
                                        "Warranty: all electronics carry a two-year manufacturer"
                                                + " warranty. Accidental damage is not covered.")));
    }

    @AfterAll
    static void unload() {
        if (chat != null) chat.close();
        if (reranker != null) reranker.close();
        if (embedder != null) embedder.close();
    }

    @Test
    @Order(1)
    void rerankerPromotesTheChunkThatActuallyAnswers() {
        List<TextSegment> shortlist = retrieve(QUESTION, 4);
        assertEquals(4, shortlist.size(), "the wide cheap net");

        // the PREMISE is model behaviour, so it is assumed, not asserted: a future embedder that
        // ranks this correctly makes the demonstration moot, not broken
        int embeddingRank = embeddingRankOfAnswer(shortlist);
        Assumptions.assumeTrue(
                embeddingRank > 0,
                "embedding retrieval already ranked the answering chunk first - nothing to fix");

        List<Double> scores = reranker.scoreAll(shortlist, QUESTION).content();
        assertEquals(shortlist.size(), scores.size(), "one score per candidate, input order");

        int rerankTop = argmax(scores);
        assertTrue(
                shortlist.get(rerankTop).text().contains(ANSWER_FACT),
                "reranked top must be the answering chunk (embedding had it at rank "
                        + embeddingRank
                        + "), was: "
                        + shortlist.get(rerankTop).text());
        assertTrue(
                scores.get(rerankTop) > scores.get(0),
                "the chunk retrieval preferred must lose the judgement: " + scores);
    }

    @Test
    @Order(2)
    void reRankingContentAggregatorNarrowsTheShortlist() {
        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .contentRetriever(retriever(4))
                        .contentAggregator(
                                ReRankingContentAggregator.builder()
                                        .scoringModel(reranker)
                                        .maxResults(2) // 4 retrieved -> 2 kept, by verdict
                                        .build())
                        .build();
        List<Content> kept = augment(augmentor, QUESTION);
        assertEquals(2, kept.size(), "the aggregator keeps maxResults");
        assertTrue(
                kept.get(0).textSegment().text().contains(ANSWER_FACT),
                "best content: " + kept.get(0).textSegment().text());
    }

    @Test
    @Order(3)
    void minScoreDropsContentsThatDoNotAnswer() {
        RetrievalAugmentor gated =
                DefaultRetrievalAugmentor.builder()
                        .contentRetriever(retriever(3))
                        .contentAggregator(
                                ReRankingContentAggregator.builder()
                                        .scoringModel(reranker)
                                        .minScore(0.5) // the verdict IS a probability: gate on it
                                        .build())
                        .build();
        // retrieval always returns its 3 nearest neighbours; nothing in the corpus is about bikes
        assertFalse(retrieve("Do you sell mountain bikes?", 3).isEmpty(), "retrieval still fires");
        assertTrue(
                augment(gated, "Do you sell mountain bikes?").isEmpty(),
                "no chunk answers an off-topic question - the gate must empty the context");
        assertFalse(augment(gated, QUESTION).isEmpty(), "the same gate keeps a real answer");
    }

    @Test
    @Order(4)
    void aiServicesAnswersFromTheRerankedContext() {
        RetrievalAugmentor augmentor =
                DefaultRetrievalAugmentor.builder()
                        .contentRetriever(retriever(4))
                        .contentAggregator(
                                ReRankingContentAggregator.builder()
                                        .scoringModel(reranker)
                                        .maxResults(1) // ONE chunk reaches the model
                                        .build())
                        .build();
        String reranked =
                AiServices.builder(Assistant.class)
                        .chatModel(chat)
                        .retrievalAugmentor(augmentor)
                        .build()
                        .answer(QUESTION);
        // one chunk reaches the model either way; only the reranked pipeline picks one that
        // CONTAINS the window, so a grounded answer proves the whole two-stage flow
        String plain =
                AiServices.builder(Assistant.class)
                        .chatModel(chat)
                        .contentRetriever(retriever(1)) // whatever embedding similarity liked most
                        .build()
                        .answer(QUESTION);
        System.out.println("PLAIN    > " + plain);
        System.out.println("RERANKED > " + reranked);
        assertTrue(reranked.contains("14"), reranked);
    }

    // ---- helpers ----

    private static ContentRetriever retriever(int maxResults) {
        return EmbeddingStoreContentRetriever.builder()
                .embeddingModel(embedder)
                .embeddingStore(store)
                .maxResults(maxResults)
                .build();
    }

    private static List<Content> augment(RetrievalAugmentor augmentor, String question) {
        UserMessage message = UserMessage.from(question);
        return augmentor
                .augment(new AugmentationRequest(message, Metadata.from(message, null, null)))
                .contents();
    }

    private static List<TextSegment> retrieve(String question, int maxResults) {
        List<TextSegment> segments = new ArrayList<>();
        store.search(
                        EmbeddingSearchRequest.builder()
                                .queryEmbedding(embedder.embed(question).content())
                                .maxResults(maxResults)
                                .build())
                .matches()
                .forEach(match -> segments.add(match.embedded()));
        return segments;
    }

    private static int embeddingRankOfAnswer(List<TextSegment> shortlist) {
        for (int i = 0; i < shortlist.size(); i++) {
            if (shortlist.get(i).text().contains(ANSWER_FACT)) return i;
        }
        return -1;
    }

    private static int argmax(List<Double> scores) {
        return IntStream.range(0, scores.size())
                .boxed()
                .max(Comparator.comparingDouble(scores::get))
                .orElseThrow();
    }
}
