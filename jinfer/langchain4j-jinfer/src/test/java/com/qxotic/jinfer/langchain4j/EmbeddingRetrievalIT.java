package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingSearchResult;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.List;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

/**
 * The composed RAG wiring over {@link JinferEmbeddingModel}: langchain4j's ingestor (splitter +
 * embedAll + store), direct store search, and the {@link ContentRetriever} used by AiServices - the
 * exact plumbing of a fully-local retrieval stack, verified end to end.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class EmbeddingRetrievalIT {

    static JinferEmbeddingModel embedder;
    static InMemoryEmbeddingStore<TextSegment> store;

    @BeforeAll
    static void ingest() {
        embedder =
                JinferEmbeddingModel.builder()
                        .modelPath(ModelFixture.QWEN3_EMBED_06B_Q8.require())
                        .build();
        store = new InMemoryEmbeddingStore<>();
        List<Document> docs =
                List.of(
                        Document.from(
                                "Refund policy: opened items can be returned within 14 days for"
                                    + " store credit only. Unopened items are refunded in full to"
                                    + " the original payment method within 30 days of purchase."),
                        Document.from(
                                "Shipping: standard delivery takes 3-5 business days. Express"
                                        + " delivery is next business day for orders placed before"
                                        + " 15:00. We ship to all EU countries and Switzerland."),
                        Document.from(
                                "Warranty: all electronics carry a two-year manufacturer warranty."
                                    + " Accidental damage is not covered; extended coverage can be"
                                    + " purchased within 30 days of delivery."));
        EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(64, 8))
                .embeddingModel(embedder)
                .embeddingStore(store)
                .build()
                .ingest(docs);
    }

    @Test
    void storeSearchFindsTheRightChunk() {
        EmbeddingSearchResult<TextSegment> hits =
                store.search(
                        EmbeddingSearchRequest.builder()
                                .queryEmbedding(
                                        embedder.embed("Can I return a product I already opened?")
                                                .content())
                                .maxResults(2)
                                .build());
        assertFalse(hits.matches().isEmpty());
        String top = hits.matches().get(0).embedded().text().toLowerCase();
        assertTrue(top.contains("refund") || top.contains("returned"), "top hit: " + top);
    }

    @Test
    void contentRetrieverServesAiServices() {
        ContentRetriever retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingModel(embedder)
                        .embeddingStore(store)
                        .maxResults(2)
                        .build();
        List<Content> contents = retriever.retrieve(Query.from("How fast is express shipping?"));
        assertEquals(2, contents.size());
        String top = contents.get(0).textSegment().text().toLowerCase();
        assertTrue(top.contains("express") || top.contains("delivery"), "top content: " + top);
    }

    @Test
    void distinctTopicsRetrieveDistinctChunks() {
        // the splitter may serve a SUB-chunk of the right document - match its topic words
        String refund = topHit("store credit for an opened box");
        String warranty = topHit("is accidental damage covered by the guarantee");
        assertTrue(
                refund.contains("refund") || refund.contains("return") || refund.contains("credit"),
                refund);
        assertTrue(
                warranty.contains("warranty")
                        || warranty.contains("damage")
                        || warranty.contains("coverage"),
                warranty);
    }

    private static String topHit(String query) {
        return store.search(
                        EmbeddingSearchRequest.builder()
                                .queryEmbedding(embedder.embed(query).content())
                                .maxResults(1)
                                .build())
                .matches()
                .get(0)
                .embedded()
                .text()
                .toLowerCase();
    }
}
