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
            ObjectProvider<ChatModelObservationConvention> observationConvention,
            ObjectProvider<com.qxotic.jinfer.media.VideoSampler> videoSampler) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.chat.model is required: a local GGUF path or a hub ref like"
                            + " hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M");
        }
        JinferChatModel.Builder builder =
                JinferChatModel.builder()
                        .model(properties.model())
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
        videoSampler.ifAvailable(builder::videoSampler); // a VideoSampler bean overrides UNIFORM
        if (properties.companions() != null) {
            properties.companions().forEach(builder::companion); // path-or-ref, like model
        }
        if (StringUtils.hasText(properties.cachedPrompts())) {
            builder.loadCachedPrompts(Path.of(properties.cachedPrompts()));
        }
        return builder.build();
    }
}
