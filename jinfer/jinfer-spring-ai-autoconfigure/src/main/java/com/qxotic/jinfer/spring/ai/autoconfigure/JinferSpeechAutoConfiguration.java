package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import java.nio.file.Path;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * Wires one {@link JinferSpeechModel} bean from {@code spring.ai.jinfer.speech.*}.
 *
 * <p>Activated by the model path, like reranking and for the same reason: Spring AI's {@code
 * spring.ai.model.*} switches name the model types it defines, and pointing this at a GGUF is the
 * unambiguous signal that speech is wanted. An app with no speech model configures nothing.
 */
@AutoConfiguration
@ConditionalOnClass(JinferSpeechModel.class)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.speech", name = "model")
@EnableConfigurationProperties(JinferSpeechProperties.class)
public class JinferSpeechAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferSpeechModel jinferSpeechModel(JinferSpeechProperties properties) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.speech.model is required: a speech GGUF as a local path or"
                            + " model ref (remixerdec/Inflect-Nano-v2-GGUF:Q8_0)");
        }
        Double speed = properties.speed();
        if (speed != null && (!(speed > 0) || Double.isInfinite(speed))) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.speech.speed must be a positive finite number, got " + speed);
        }
        int maxInputChars = properties.maxInputChars();
        if (maxInputChars < 0) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.speech.max-input-chars must be 0 (use the adapter default) or"
                            + " at least 1, got "
                            + maxInputChars);
        }
        JinferSpeechModel.Builder builder = JinferSpeechModel.builder();
        if (properties.model().contains("://")) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.speech.model is a URL; download it first and configure its"
                            + " local path");
        } else if (ModelStore.isRef(properties.model())) {
            builder.model(properties.model());
        } else {
            builder.modelPath(Path.of(properties.model()));
        }
        if (properties.companions() != null) {
            properties
                    .companions()
                    .forEach(
                            (capability, value) -> {
                                if (!StringUtils.hasText(value)) {
                                    throw new IllegalStateException(
                                            "spring.ai.jinfer.speech.companions."
                                                    + capability
                                                    + " must not be blank");
                                }
                                if (value.contains("://")) {
                                    throw new IllegalStateException(
                                            "spring.ai.jinfer.speech.companions."
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
        if (speed != null) builder.speed(speed);
        if (maxInputChars > 0) builder.maxInputChars(maxInputChars);
        return builder.build();
    }
}
