package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferTranscriptionModel;
import io.micrometer.observation.ObservationRegistry;
import org.springframework.ai.audio.transcription.observation.TranscriptionModelObservationConvention;
import org.springframework.beans.factory.ObjectProvider;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.util.StringUtils;

/**
 * Wires one {@link JinferTranscriptionModel} bean from {@code spring.ai.jinfer.transcription.*}.
 *
 * <p>Active when a transcription model is configured and {@code
 * spring.ai.model.audio.transcription} selects jinfer or no provider, as Spring AI's own
 * transcription providers are; another provider selected switches it off. An app that does not
 * transcribe configures nothing.
 */
@AutoConfiguration
@ConditionalOnClass(JinferTranscriptionModel.class)
@ConditionalOnProperty(
        name = "spring.ai.model.audio.transcription",
        havingValue = "jinfer",
        matchIfMissing = true)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.transcription", name = "model")
@EnableConfigurationProperties(JinferTranscriptionProperties.class)
public class JinferTranscriptionAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferTranscriptionModel jinferTranscriptionModel(
            JinferTranscriptionProperties properties,
            ObjectProvider<ObservationRegistry> observationRegistry,
            ObjectProvider<TranscriptionModelObservationConvention> observationConvention) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.transcription.model is required: a transcription GGUF as a"
                            + " local path or model ref"
                            + " (mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf)");
        }
        JinferTranscriptionModel.Builder builder = JinferTranscriptionModel.builder();
        Sources.model(
                "spring.ai.jinfer.transcription.model",
                properties.model(),
                builder::model,
                builder::modelPath);
        observationRegistry.ifAvailable(builder::observationRegistry);
        observationConvention.ifAvailable(builder::observationConvention);
        return builder.build();
    }
}
