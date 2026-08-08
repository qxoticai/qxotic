package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.embedding.observation.EmbeddingModelObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * Wires one {@link JinferEmbeddingModel} bean from {@code spring.ai.jinfer.embedding.*}. Unlike
 * chat (the starter's default capability), embedding requires explicit selection - a chat-only app
 * must not be forced to point at an embedding GGUF.
 */
@AutoConfiguration
@ConditionalOnClass(JinferEmbeddingModel.class)
@ConditionalOnProperty(name = "spring.ai.model.embedding", havingValue = "jinfer")
@EnableConfigurationProperties(JinferEmbeddingProperties.class)
public class JinferEmbeddingAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferEmbeddingModel jinferEmbeddingModel(
            JinferEmbeddingProperties properties,
            ObjectProvider<ObservationRegistry> observationRegistry,
            ObjectProvider<EmbeddingModelObservationConvention> observationConvention) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.embedding.model is required: an embedding GGUF (e.g."
                            + " Qwen3-Embedding) as a local path or a hub ref");
        }
        JinferEmbeddingModel.Builder builder =
                JinferEmbeddingModel.builder()
                        .model(properties.model())
                        .contextLength(properties.contextLength());
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        return builder.build();
    }
}
