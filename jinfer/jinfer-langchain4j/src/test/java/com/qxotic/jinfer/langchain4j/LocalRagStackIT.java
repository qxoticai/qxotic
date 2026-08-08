package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.JsonCodec;
import com.qxotic.jinfer.testkit.ModelFixture;
import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.SystemMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.request.ResponseFormat;
import dev.langchain4j.model.chat.request.ResponseFormatType;
import dev.langchain4j.model.chat.request.json.JsonIntegerSchema;
import dev.langchain4j.model.chat.request.json.JsonObjectSchema;
import dev.langchain4j.model.chat.request.json.JsonSchema;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.query.Query;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingSearchRequest;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.MethodOrderer;
import org.junit.jupiter.api.Order;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.TestMethodOrder;

/**
 * The fully-local RAG stack - chat + embeddings together in one JVM - as a graded progression of
 * real-life setups, simplest first:
 *
 * <ol>
 *   <li>FAQ bot: one question, answered from the docs (AiServices + retriever).
 *   <li>Support thread: follow-up questions, retrieval + conversation memory.
 *   <li>Ticket triage: embeddings route the request, chat answers at the routed desk.
 *   <li>Back office: retrieved policy text extracted to schema-guaranteed JSON.
 *   <li>Production desk: the support scaffold prepaid via withCachedPrompt, each request paying
 *       only retrieval + the question.
 * </ol>
 *
 * <p>The corpus facts (store credit, next business day, two-year warranty) exist ONLY in the
 * retrieved chunks, so a grounded answer proves the retrieval-to-chat flow, not model memory.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@TestMethodOrder(MethodOrderer.OrderAnnotation.class)
class LocalRagStackIT {

    static JinferEmbeddingModel embedder;
    static JinferChatModel chat;
    static InMemoryEmbeddingStore<TextSegment> store;
    static ContentRetriever retriever;

    interface Assistant {
        String answer(String question);
    }

    @BeforeAll
    static void buildStack() {
        embedder =
                JinferEmbeddingModel.builder()
                        .modelPath(ModelFixture.QWEN3_EMBED_06B_Q8.require())
                        .build();
        chat =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.GEMMA4_E2B_Q8.require())
                        .contextLength(4096)
                        .temperature(0.0)
                        .maxOutputTokens(160)
                        .build();
        store = new InMemoryEmbeddingStore<>();
        EmbeddingStoreIngestor.builder()
                .documentSplitter(DocumentSplitters.recursive(128, 16))
                .embeddingModel(embedder)
                .embeddingStore(store)
                .build()
                .ingest(
                        List.of(
                                Document.from(
                                        "Refund policy: opened items can be returned within 14 days"
                                            + " for store credit only. Unopened items are refunded"
                                            + " in full within 30 days of purchase."),
                                Document.from(
                                        "Shipping: standard delivery takes 3-5 business days."
                                            + " Express delivery arrives the next business day for"
                                            + " orders placed before 15:00."),
                                Document.from(
                                        "Warranty: all electronics carry a two-year manufacturer"
                                                + " warranty. Accidental damage is not covered.")));
        retriever =
                EmbeddingStoreContentRetriever.builder()
                        .embeddingModel(embedder)
                        .embeddingStore(store)
                        .maxResults(2)
                        .build();
    }

    @AfterAll
    static void unload() {
        if (chat != null) chat.close();
        if (embedder != null) embedder.close();
    }

    @Test
    @Order(1)
    void faqBotAnswersOneQuestionFromTheDocs() {
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chat)
                        .contentRetriever(retriever)
                        .build();
        String answer =
                assistant.answer(
                        "I opened the box already - can I still return it, and what"
                                + " do I get back?");
        // "store credit" exists ONLY in the retrieved chunk: a grounded answer proves the flow
        assertTrue(answer.toLowerCase().contains("store credit"), answer);
    }

    @Test
    @Order(2)
    void supportThreadFollowsUpWithMemory() {
        Assistant assistant =
                AiServices.builder(Assistant.class)
                        .chatModel(chat)
                        .contentRetriever(retriever)
                        .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                        .build();
        String first = assistant.answer("How fast is express shipping?");
        assertTrue(first.toLowerCase().contains("next business day"), first);
        // the follow-up only makes sense with memory; the cutoff fact rides retrieval again
        String second =
                assistant.answer("And what is the order cutoff time for that next-day option?");
        assertTrue(second.contains("15:00") || second.toLowerCase().contains("3 pm"), second);
    }

    @Test
    @Order(3)
    void ticketTriageRoutesByEmbeddingThenChatAnswers() {
        Map<String, Embedding> routes =
                Map.of(
                        "returns",
                                embedder.embed("refunds, returns, store credit, opened items")
                                        .content(),
                        "shipping",
                                embedder.embed("delivery, parcels, shipping speed, couriers")
                                        .content(),
                        "warranty",
                                embedder.embed("guarantees, defects, repairs, coverage").content());
        Embedding incoming = embedder.embed("my parcel still has not arrived").content();
        String route =
                routes.entrySet().stream()
                        .max(Comparator.comparingDouble(e -> cosine(incoming, e.getValue())))
                        .orElseThrow()
                        .getKey();
        assertEquals("shipping", route); // the embedding decision is exact
        ChatResponse routed =
                chat.chat(
                        ChatRequest.builder()
                                .messages(
                                        SystemMessage.from(
                                                "You are the "
                                                        + route
                                                        + " desk. Answer in one sentence."),
                                        UserMessage.from(
                                                "My parcel has not arrived. What are the normal"
                                                        + " delivery times?"))
                                .build());
        assertFalse(routed.aiMessage().text().isBlank());
    }

    @Test
    @Order(4)
    void backOfficeExtractsPolicyFactsAsSchemaJson() {
        // retrieve the refund chunk, then extract structured facts under a JSON-schema grammar -
        // the grammar GUARANTEES the shape; retrieval grounds the values
        String context =
                store.search(
                                EmbeddingSearchRequest.builder()
                                        .queryEmbedding(
                                                embedder.embed("return window for opened items")
                                                        .content())
                                        .maxResults(1)
                                        .build())
                        .matches()
                        .get(0)
                        .embedded()
                        .text();
        ChatResponse r =
                chat.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Policy text: \""
                                                        + context
                                                        + "\"\nExtract the return window in days"
                                                        + " for opened items."))
                                .responseFormat(
                                        ResponseFormat.builder()
                                                .type(ResponseFormatType.JSON)
                                                .jsonSchema(
                                                        JsonSchema.builder()
                                                                .name("policy")
                                                                .rootElement(
                                                                        JsonObjectSchema.builder()
                                                                                .addProperty(
                                                                                        "days",
                                                                                        new JsonIntegerSchema())
                                                                                .required("days")
                                                                                .build())
                                                                .build())
                                                .build())
                                .build());
        Object parsed = JsonCodec.parse(r.aiMessage().text()); // the grammar guarantees valid JSON
        assertTrue(
                parsed instanceof Map<?, ?> m && Long.valueOf(14L).equals(m.get("days")),
                r.aiMessage().text());
    }

    @Test
    @Order(5)
    void productionDeskPrepaidScaffoldPlusRetrieval() {
        // the support scaffold is prepaid once; each request pays only retrieval + the question
        JinferChatModel support =
                chat.withCachedPrompt(
                        List.of(
                                SystemMessage.from(
                                        "You are a support agent. Answer strictly from the provided"
                                                + " policy excerpts, in one short sentence.")),
                        List.of());
        String context =
                retriever.retrieve(Query.from("warranty length")).get(0).textSegment().text();
        ChatResponse r =
                support.chat(
                        ChatRequest.builder()
                                .messages(
                                        UserMessage.from(
                                                "Policy: \""
                                                        + context
                                                        + "\"\nHow long is the warranty on"
                                                        + " electronics?"))
                                .build());
        assertTrue(r.aiMessage().text().toLowerCase().contains("two"), r.aiMessage().text());
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
