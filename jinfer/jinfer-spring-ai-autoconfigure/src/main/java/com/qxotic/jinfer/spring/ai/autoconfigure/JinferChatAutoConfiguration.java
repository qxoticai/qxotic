package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.codecs.VideoSampler;
import com.qxotic.jinfer.hub.ModelStore;
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
            ObjectProvider<VideoSampler> videoSampler) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.chat.model is required: a local GGUF path, or a model ref"
                            + " (hf.co/unsloth/gemma-4-E2B-it-GGUF:Q4_K_M)");
        }
        JinferChatModel.Builder builder =
                JinferChatModel.builder()
                        .retainSessions(properties.retainedSessions())
                        .options(properties.toOptions())
                        .speculationDepth(properties.speculationDepth());
        if (properties.contextLength() != null) builder.contextLength(properties.contextLength());
        if (properties.model().contains("://")) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.chat.model is a URL; download it first and configure its"
                            + " local path");
        } else if (ModelStore.isRef(properties.model())) {
            builder.model(properties.model());
        } else {
            builder.modelPath(Path.of(properties.model())); // local path: the explicit door
        }
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        videoSampler.ifAvailable(builder::videoSampler); // a VideoSampler bean overrides UNIFORM
        if (properties.companions() != null) {
            properties
                    .companions()
                    .forEach(
                            (capability, value) -> {
                                if (!StringUtils.hasText(value)) {
                                    throw new IllegalStateException(
                                            "spring.ai.jinfer.chat.companions."
                                                    + capability
                                                    + " must not be blank");
                                }
                                if (value.contains("://")) {
                                    throw new IllegalStateException(
                                            "spring.ai.jinfer.chat.companions."
                                                    + capability
                                                    + " is a URL; download it first and configure"
                                                    + " its local path");
                                }
                                if (ModelStore.isRef(value)) {
                                    builder.companion(capability, value);
                                } else {
                                    builder.companionPath(capability, Path.of(value));
                                }
                            });
        }
        if (StringUtils.hasText(properties.promptCache())) {
            builder.promptCache(Path.of(properties.promptCache()));
        }
        return builder.build();
    }
}
