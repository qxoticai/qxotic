package com.qxotic.jinfer.spring.ai.autoconfigure;

import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import org.springframework.boot.autoconfigure.AutoConfiguration;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnMissingBean;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;

/**
 * Wires one {@link JinferSpeechModel} bean from {@code spring.ai.jinfer.speech.*}.
 *
 * <p>Chat, embedding and reranking were auto-configured and speech was not, so a Boot application
 * got three of jinfer's four model types for free and had to hand-wire the fourth.
 *
 * <p>Activated by the model path, like reranking and for the same reason: Spring AI's {@code
 * spring.ai.model.*} switches name the model types it defines, and pointing this at a GGUF is the
 * unambiguous signal that speech is wanted. An app with no speech model configures nothing.
 *
 * <p>The bean owns an arena and is {@link AutoCloseable}; Spring calls {@code close()} at context
 * shutdown, which frees the weights deterministically rather than at a GC a native-heavy JVM may
 * never run.
 */
@AutoConfiguration
@ConditionalOnClass(JinferSpeechModel.class)
@ConditionalOnProperty(prefix = "spring.ai.jinfer.speech", name = "model")
@EnableConfigurationProperties(JinferSpeechProperties.class)
public class JinferSpeechAutoConfiguration {

    @Bean
    @ConditionalOnMissingBean
    public JinferSpeechModel jinferSpeechModel(JinferSpeechProperties properties) {
        JinferSpeechModel.Builder builder = JinferSpeechModel.builder().model(properties.model());
        // 0 means "leave the model's own default alone" - passing it through would override the
        // port's choice with a meaningless value
        if (properties.speed() > 0) builder.speed(properties.speed());
        if (properties.maxInputChars() > 0) builder.maxInputChars(properties.maxInputChars());
        return builder.build();
    }
}
