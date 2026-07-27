package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import io.micrometer.observation.ObservationRegistry;
import java.nio.file.Path;
import org.springframework.ai.chat.observation.ChatModelObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/** Wires one {@link JinferChatModel} bean from {@code spring.ai.jinfer.chat.*} properties. */
@AutoConfiguration
@ConditionalOnClass(JinferChatModel.class)
@ConditionalOnProperty(name = "spring.ai.model.chat", havingValue = "jinfer", matchIfMissing = true)
@EnableConfigurationProperties(JinferChatProperties.class)
public class JinferChatAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferChatModel jinferChatModel(
            JinferChatProperties properties,
            ObjectProvider<ObservationRegistry> observationRegistry,
            ObjectProvider<ChatModelObservationConvention> observationConvention) {
        if (!StringUtils.hasText(properties.modelPath())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.chat.model-path is required: point it at a local GGUF file");
        }
        JinferChatModel.Builder builder =
                JinferChatModel.builder()
                        .modelPath(Path.of(properties.modelPath()))
                        .contextLength(properties.contextLength())
                        .cachedSessions(properties.cachedSessions())
                        .temperature(properties.temperature())
                        .topP(properties.topP())
                        .maxTokens(properties.maxTokens())
                        .seed(properties.seed())
                        .thinking(properties.thinking())
                        .timeout(properties.timeout());
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        if (StringUtils.hasText(properties.mediaProjector())) {
            builder.mediaProjector(Path.of(properties.mediaProjector()));
        }
        if (StringUtils.hasText(properties.cachedPrompts())) {
            builder.loadCachedPrompts(Path.of(properties.cachedPrompts()));
        }
        return builder.build();
    }
}
