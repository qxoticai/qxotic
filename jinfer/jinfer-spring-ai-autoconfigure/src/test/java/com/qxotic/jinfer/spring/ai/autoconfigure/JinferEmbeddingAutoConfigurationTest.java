package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/** Context behavior, no model needed: selection, required properties, property binding. */
class JinferEmbeddingAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(
                            AutoConfigurations.of(JinferEmbeddingAutoConfiguration.class));

    @Test
    void dormantByDefault() {
        // chat-only apps must not be forced to point at an embedding GGUF
        runner.run(
                context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).doesNotHaveBean(JinferEmbeddingModel.class);
                });
    }

    @Test
    void modelPathIsRequiredWhenSelected() {
        runner.withPropertyValues("spring.ai.model.embedding=jinfer")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasMessageContaining("spring.ai.jinfer.embedding.model-path");
                        });
    }

    @Test
    void backsOffWhenAnotherProviderIsSelected() {
        runner.withPropertyValues(
                        "spring.ai.model.embedding=ollama",
                        "spring.ai.jinfer.embedding.model-path=/x.gguf")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).doesNotHaveBean(JinferEmbeddingModel.class);
                        });
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.embedding.model-path=/emb.gguf",
                        "spring.ai.jinfer.embedding.context-length=1024")
                .run(
                        context -> {
                            JinferEmbeddingProperties p =
                                    context.getBean(JinferEmbeddingProperties.class);
                            assertThat(p.modelPath()).isEqualTo("/emb.gguf");
                            assertThat(p.contextLength()).isEqualTo(1024);
                        });
    }

    @EnableConfigurationProperties(JinferEmbeddingProperties.class)
    static class PropsOnly {}
}
