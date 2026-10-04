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
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Conditional;
import org.springframework.util.StringUtils;

/**
 * Wires one {@link JinferChatModel} bean from {@code spring.ai.jinfer.chat.*} properties.
 *
 * <p>Active when {@code spring.ai.model.chat=jinfer} selects jinfer explicitly, or when no chat
 * provider is selected and {@code spring.ai.jinfer.chat.model} is set:
 *
 * <pre>{@code
 * spring.ai.jinfer.chat.model=unsloth/gemma-4-E2B-it-GGUF:Q4_K_M
 * spring.ai.jinfer.chat.temperature=0.7
 * }</pre>
 */
@AutoConfiguration
@ConditionalOnClass(JinferChatModel.class)
@Conditional(JinferChatSelected.class)
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
                            + " (unsloth/gemma-4-E2B-it-GGUF:Q4_K_M)");
        }
        validateSampling(properties);
        JinferChatModel.Builder builder =
                JinferChatModel.builder()
                        .retainSessions(properties.retainedSessions())
                        .options(properties.toOptions())
                        .speculationDepth(properties.speculationDepth());
        if (properties.contextCapacity() != null)
            builder.contextCapacity(properties.contextCapacity());
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

    /**
     * Range checks on the generation properties, naming the property: run before the model
     * resolves, so a typo fails the boot in milliseconds instead of after a download, and never
     * reaches the first request as a bare sampler message.
     */
    private static void validateSampling(JinferChatProperties p) {
        Double temperature = p.temperature();
        if (temperature != null && !(Double.isFinite(temperature) && temperature >= 0))
            throw invalid("temperature", "must be >= 0", temperature);
        if (p.topP() != null && !(p.topP() > 0 && p.topP() <= 1))
            throw invalid("top-p", "must be within (0, 1]", p.topP());
        if (p.topK() != null && p.topK() < 0)
            throw invalid("top-k", "must be >= 0 (0 disables it)", p.topK());
        if (p.minP() != null && !(p.minP() >= 0 && p.minP() <= 1))
            throw invalid("min-p", "must be within [0, 1]", p.minP());
        if (p.maxReasoningTokens() != null && p.maxReasoningTokens() < -1)
            throw invalid(
                    "max-reasoning-tokens",
                    "must be -1 (uncapped) or >= 0",
                    p.maxReasoningTokens());
        if (p.timeout() != null && p.timeout().isNegative())
            throw invalid("timeout", "must not be negative", p.timeout());
    }

    private static IllegalStateException invalid(String property, String rule, Object value) {
        return new IllegalStateException(
                "spring.ai.jinfer.chat." + property + " " + rule + ": " + value);
    }
}
