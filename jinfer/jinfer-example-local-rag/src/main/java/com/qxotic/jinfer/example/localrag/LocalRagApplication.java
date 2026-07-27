package com.qxotic.jinfer.example.localrag;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import java.util.ArrayList;
import java.util.List;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.advisor.RetrievalAugmentationAdvisor;
import org.springframework.ai.rag.retrieval.search.VectorStoreDocumentRetriever;
import org.springframework.ai.transformer.splitter.TokenTextSplitter;
import org.springframework.ai.vectorstore.SimpleVectorStore;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;

/**
 * Fully-local RAG in one JVM: Qwen3-Embedding vectors + a chat GGUF, no server, no API key. A tiny
 * support corpus whose facts exist ONLY in the documents (store credit, next business day, two-year
 * warranty) - so a grounded answer proves the retrieve-then-generate flow, not model memory. Run:
 * {@code mvn spring-boot:run}.
 */
@SpringBootApplication
public class LocalRagApplication {

    /** Facts that exist ONLY here - retrieval is the only way the model can know them. */
    static final List<Document> CORPUS =
            List.of(
                    new Document(
                            "AcmeCloud returns: returns are accepted within 30 days of delivery."
                                    + " Refunds are issued as store credit to the customer's"
                                    + " account, never back to the original payment method."),
                    new Document(
                            "AcmeCloud shipping: orders placed before 3pm ship the next business"
                                    + " day. Orders placed on Saturday or Sunday ship on Monday."),
                    new Document(
                            "AcmeCloud warranty: every appliance carries a two-year warranty"
                                    + " covering parts and labor, starting on the delivery date."),
                    new Document(
                            "AcmeCloud support hours: the help desk answers Monday to Friday,"
                                    + " 9am to 6pm CET, in English and German."));

    static final List<String> QUESTIONS =
            List.of(
                    "How will I get my refund?",
                    "I ordered something on Saturday - when does it ship?",
                    "How long is the warranty on my appliance?");

    public static void main(String[] args) {
        SpringApplication.run(LocalRagApplication.class, args);
    }

    @Bean
    CommandLineRunner demo(JinferChatModel chat, JinferEmbeddingModel embeddings) {
        return args -> run(chat, embeddings);
    }

    /** Ingest the corpus, then answer the questions retrieval-grounded. Returns the answers. */
    static List<String> run(JinferChatModel chat, JinferEmbeddingModel embeddings) {
        long t0 = System.nanoTime();
        VectorStore store = SimpleVectorStore.builder(embeddings).build();
        store.add(new TokenTextSplitter().split(CORPUS));
        System.out.printf(
                ">>> ingested %d documents in %.1fs%n",
                CORPUS.size(), (System.nanoTime() - t0) / 1e9);

        ChatClient rag =
                ChatClient.builder(chat)
                        .defaultAdvisors(
                                RetrievalAugmentationAdvisor.builder()
                                        .documentRetriever(
                                                VectorStoreDocumentRetriever.builder()
                                                        .vectorStore(store)
                                                        .topK(2)
                                                        .build())
                                        .build())
                        .build();

        List<String> answers = new ArrayList<>();
        for (String question : QUESTIONS) {
            String answer = rag.prompt(question).call().content();
            answers.add(answer);
            System.out.printf(">>> Q: %s%n>>> A: %s%n", question, answer.strip());
        }
        return answers;
    }
}
