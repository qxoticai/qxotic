package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.audio.tts.observation.TextToSpeechModelObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
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
 * <p>Active when a speech model is configured and {@code spring.ai.model.audio.speech} selects
 * jinfer or no provider, as Spring AI's own speech providers are; another provider selected
 * switches it off. An app with no speech model configures nothing.
 */
@AutoConfiguration
@ConditionalOnClass(JinferSpeechModel.class)
@ConditionalOnProperty(
        name = "spring.ai.model.audio.speech",
        havingValue = "jinfer",
        matchIfMissing = true)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.speech", name = "model")
@EnableConfigurationProperties(JinferSpeechProperties.class)
public class JinferSpeechAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferSpeechModel jinferSpeechModel(
            JinferSpeechProperties properties,
            ObjectProvider<ObservationRegistry> observationRegistry,
            ObjectProvider<TextToSpeechModelObservationConvention> observationConvention) {
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
        Sources.model(
                "spring.ai.jinfer.speech.model",
                properties.model(),
                builder::model,
                builder::modelPath);
        Sources.companions(
                "spring.ai.jinfer.speech",
                properties.companions(),
                builder::companion,
                builder::companionPath);
        if (speed != null) builder.speed(speed);
        if (maxInputChars > 0) builder.maxInputChars(maxInputChars);
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        return builder.build();
    }
}
