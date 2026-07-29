package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor;
import java.nio.file.Path;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * Wires one {@link JinferDocumentPostProcessor} bean from {@code spring.ai.jinfer.rerank.*}, to
 * drop into a {@code RetrievalAugmentationAdvisor}'s post-retrieval stage.
 *
 * <p>Selection is the model path itself, not a {@code spring.ai.model.*} switch: Spring AI defines
 * capability keys for the model types it knows (chat, embedding, image, ...) and has no reranking
 * model type at all - reranking is a RAG pipeline step. Inventing {@code spring.ai.model.rerank}
 * would claim a contract no other provider participates in, so the bean simply appears once you
 * point it at a reranker GGUF, and a rerank-less app configures nothing.
 */
@AutoConfiguration
@ConditionalOnClass(JinferDocumentPostProcessor.class)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.rerank", name = "model-path")
@EnableConfigurationProperties(JinferRerankProperties.class)
public class JinferRerankAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferDocumentPostProcessor jinferDocumentPostProcessor(
            JinferRerankProperties properties) {
        JinferDocumentPostProcessor.Builder builder =
                JinferDocumentPostProcessor.builder()
                        .modelPath(Path.of(properties.modelPath()))
                        .contextLength(properties.contextLength())
                        .topK(properties.topK())
                        .minScore(properties.minScore());
        if (StringUtils.hasText(properties.instruction())) {
            builder.instruction(properties.instruction());
        }
        return builder.build();
    }
}
