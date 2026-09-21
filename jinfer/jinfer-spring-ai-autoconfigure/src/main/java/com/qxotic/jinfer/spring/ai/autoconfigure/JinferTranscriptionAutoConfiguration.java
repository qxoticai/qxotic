package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.hub.ModelStore;
import com.qxotic.jinfer.spring.ai.JinferTranscriptionModel;
import java.nio.file.Path;
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
 * <p>Activated by the model path, like speech and for the same reason: pointing this at a GGUF is
 * the unambiguous signal that transcription is wanted. An app that does not transcribe configures
 * nothing.
 */
@AutoConfiguration
@ConditionalOnClass(JinferTranscriptionModel.class)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.transcription", name = "model")
@EnableConfigurationProperties(JinferTranscriptionProperties.class)
public class JinferTranscriptionAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferTranscriptionModel jinferTranscriptionModel(
            JinferTranscriptionProperties properties) {
        if (!StringUtils.hasText(properties.model())) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.transcription.model is required: a transcription GGUF as a"
                            + " local path or model ref"
                            + " (mudler/parakeet-cpp-gguf/tdt-0.6b-v3-q8_0.gguf)");
        }
        JinferTranscriptionModel.Builder builder = JinferTranscriptionModel.builder();
        if (properties.model().contains("://")) {
            throw new IllegalStateException(
                    "spring.ai.jinfer.transcription.model is a URL; download it first and configure"
                            + " its local path");
        } else if (ModelStore.isRef(properties.model())) {
            builder.model(properties.model());
        } else {
            builder.modelPath(Path.of(properties.model()));
        }
        return builder.build();
    }
}
