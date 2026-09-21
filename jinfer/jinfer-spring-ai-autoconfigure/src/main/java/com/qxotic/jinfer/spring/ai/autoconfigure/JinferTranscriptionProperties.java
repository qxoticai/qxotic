package com.qxotic.jinfer.spring.ai.autoconfigure;

import org.springframework.boot.context.properties.ConfigurationProperties;

/**
 * Configuration properties for jinfer speech-to-text, bound under {@code
 * spring.ai.jinfer.transcription} (constructor binding).
 *
 * @param model the transcription GGUF (e.g. a Parakeet model) as a local path or model ref;
 *     configuring it is what activates the model, the same rule the speech properties use
 */
@ConfigurationProperties("spring.ai.jinfer.transcription")
public record JinferTranscriptionProperties(String model) {}
