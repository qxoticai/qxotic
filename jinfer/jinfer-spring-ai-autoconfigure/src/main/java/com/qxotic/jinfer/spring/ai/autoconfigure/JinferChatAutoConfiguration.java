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
        if (!StringUtils.hasText(properties.getModelPath())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.chat.model-path is required: point it at a local GGUF file");
        }
        JinferChatModel.Builder builder =
                JinferChatModel.builder()
                        .modelPath(Path.of(properties.getModelPath()))
                        .contextLength(properties.getContextLength())
                        .temperature(properties.getTemperature())
                        .topP(properties.getTopP())
                        .maxTokens(properties.getMaxTokens())
                        .seed(properties.getSeed())
                        .thinking(properties.getThinking())
                        .timeout(properties.getTimeout());
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        if (StringUtils.hasText(properties.getMediaProjector())) {
            builder.mediaProjector(Path.of(properties.getMediaProjector()));
        }
        if (StringUtils.hasText(properties.getCachedPrompts())) {
            builder.loadCachedPrompts(Path.of(properties.getCachedPrompts()));
        }
        return builder.build();
    }
}
