package com.qxotic.jinfer.example.localrag;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.JinferChatOptions;
import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import com.qxotic.jinfer.testkit.TestModels;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The fully-local RAG stack end to end: Qwen3-Embedding 0.6B for vectors, LFM2.5 8B for chat. The
 * answers' facts (store credit, next business day, two-year warranty) exist ONLY in the corpus, so
 * containing them proves retrieval, not model memory. Model-gated via {@link TestModels}. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl jinfer-example-local-rag}
 */
@Tag("integration")
class LocalRagIT {

    @Test
    void answersAreGroundedInTheCorpus() {
        try (JinferEmbeddingModel embeddings =
                        JinferEmbeddingModel.builder()
                                .modelPath(
                                        TestModels.require(
                                                "hf.co/Qwen/Qwen3-Embedding-0.6B-GGUF:Q8_0"))
                                .build();
                JinferChatModel chat =
                        JinferChatModel.builder()
                                .modelPath(
                                        TestModels.require(
                                                "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0"))
                                .contextLength(4096)
                                .options(JinferChatOptions.builder().maxTokens(256).build())
                                .build()) {

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
}
