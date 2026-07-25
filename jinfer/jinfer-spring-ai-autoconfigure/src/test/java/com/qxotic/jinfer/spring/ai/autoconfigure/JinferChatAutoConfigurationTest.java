package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/** Context behavior, no model needed: selection, required properties, property binding. */
class JinferChatAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(AutoConfigurations.of(JinferChatAutoConfiguration.class));

    @Test
    void modelPathIsRequired() {
        runner.run(
                context -> {
                    assertThat(context).hasFailed();
                    assertThat(context.getStartupFailure())
                            .hasMessageContaining("spring.ai.jinfer.chat.model-path");
                });
    }

    @Test
    void backsOffWhenAnotherProviderIsSelected() {
        runner.withPropertyValues(
                        "spring.ai.model.chat=ollama", "spring.ai.jinfer.chat.model-path=/x.gguf")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).doesNotHaveBean(JinferChatModel.class);
                        });
    }

    @Test
    void propertiesBind() {
        // binding only: the model bean needs a real GGUF, so this runs without the auto-config
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model-path=/x.gguf",
                        "spring.ai.jinfer.chat.media-projector=/mmproj.gguf",
                        "spring.ai.jinfer.chat.cached-prompts=/personas.jkv",
                        "spring.ai.jinfer.chat.context-length=8192",
                        "spring.ai.jinfer.chat.temperature=0.7",
                        "spring.ai.jinfer.chat.top-p=0.9",
                        "spring.ai.jinfer.chat.max-tokens=512",
                        "spring.ai.jinfer.chat.seed=7",
                        "spring.ai.jinfer.chat.thinking=false",
                        "spring.ai.jinfer.chat.timeout=30s")
                .run(
                        context -> {
                            JinferChatProperties p = context.getBean(JinferChatProperties.class);
                            assertThat(p.getModelPath()).isEqualTo("/x.gguf");
                            assertThat(p.getMediaProjector()).isEqualTo("/mmproj.gguf");
                            assertThat(p.getCachedPrompts()).isEqualTo("/personas.jkv");
                            assertThat(p.getContextLength()).isEqualTo(8192);
                            assertThat(p.getTemperature()).isEqualTo(0.7);
                            assertThat(p.getTopP()).isEqualTo(0.9);
                            assertThat(p.getMaxTokens()).isEqualTo(512);
                            assertThat(p.getSeed()).isEqualTo(7L);
                            assertThat(p.getThinking()).isFalse();
                            assertThat(p.getTimeout()).hasSeconds(30);
                        });
    }

    @EnableConfigurationProperties(JinferChatProperties.class)
    static class PropsOnly {}
}
