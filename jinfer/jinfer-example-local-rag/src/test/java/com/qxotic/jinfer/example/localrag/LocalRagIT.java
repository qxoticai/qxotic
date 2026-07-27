package com.qxotic.jinfer.example.localrag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import com.qxotic.jinfer.testkit.ModelFixture;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The fully-local RAG stack end to end: Qwen3-Embedding 0.6B for vectors, LFM2.5 8B for chat. The
 * answers' facts (store credit, next business day, two-year warranty) exist ONLY in the corpus, so
 * containing them proves retrieval, not model memory. Model-gated via {@link ModelFixture}. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-example-local-rag}
 */
@Tag("integration")
class LocalRagIT {

    @Test
    void answersAreGroundedInTheCorpus() {
        JinferEmbeddingModel embeddings =
                JinferEmbeddingModel.builder()
                        .modelPath(ModelFixture.QWEN3_EMBED_06B_Q8.require())
                        .build();
        JinferChatModel chat =
                JinferChatModel.builder()
                        .modelPath(ModelFixture.LFM25_8B_Q8.require())
                        .contextLength(4096)
                        .maxTokens(256)
                        .build();

        List<String> answers = LocalRagApplication.run(chat, embeddings);

        assertEquals(LocalRagApplication.QUESTIONS.size(), answers.size());
        String refund = answers.get(0).toLowerCase();
        assertTrue(refund.contains("store credit"), "refund answer ungrounded: " + refund);
        String shipping = answers.get(1).toLowerCase();
        assertTrue(
                shipping.contains("next business day") || shipping.contains("monday"),
                "shipping answer ungrounded: " + shipping);
        String warranty = answers.get(2).toLowerCase();
        assertTrue(
                warranty.contains("two-year") || warranty.contains("two year"),
                "warranty answer ungrounded: " + warranty);
    }
}
