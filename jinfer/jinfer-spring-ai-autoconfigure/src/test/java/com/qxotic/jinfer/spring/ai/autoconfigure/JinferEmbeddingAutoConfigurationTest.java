package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import com.qxotic.jinfer.spring.ai.JinferEmbeddingModel;
import org.junit.jupiter.api.Test;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

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
    void modelIsRequiredWhenSelected() {
        runner.withPropertyValues("spring.ai.model.embedding=jinfer")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasMessageContaining("spring.ai.jinfer.embedding.model");
                        });
    }

    @Test
    void modelUrlIsRejectedBeforeResolution() {
        runner.withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=https://example.org/model.gguf")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining("download it first");
                        });
    }

    @Test
    void backsOffWhenAnotherProviderIsSelected() {
        runner.withPropertyValues(
                        "spring.ai.model.embedding=ollama",
                        "spring.ai.jinfer.embedding.model=/x.gguf")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).doesNotHaveBean(JinferEmbeddingModel.class);
                        });
    }

    @Test
    void backsOffForAnInterfaceReturningUserBeanWithoutModelProperties() {
        runner.withUserConfiguration(UserEmbeddingConfiguration.class)
                .withPropertyValues("spring.ai.model.embedding=jinfer")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(EmbeddingModel.class);
                            assertThat(context).doesNotHaveBean("jinferEmbeddingModel");
                            assertThat(context.getBean(EmbeddingModel.class))
                                    .isSameAs(context.getBean("userEmbeddingModel"));
                        });
    }

    @Test
    void backsOffForAConcreteUserBeanWithoutModelProperties() {
        JinferEmbeddingModel custom = mock(JinferEmbeddingModel.class);
        runner.withBean("userEmbeddingModel", JinferEmbeddingModel.class, () -> custom)
                .withPropertyValues("spring.ai.model.embedding=jinfer")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(EmbeddingModel.class);
                            assertThat(context).doesNotHaveBean("jinferEmbeddingModel");
                            assertThat(context.getBean(EmbeddingModel.class)).isSameAs(custom);
                        });
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.embedding.model=/emb.gguf",
                        "spring.ai.jinfer.embedding.companions.media=/mmproj.gguf",
                        "spring.ai.jinfer.embedding.context-capacity=1024",
                        "spring.ai.jinfer.embedding.dimensions=256")
                .run(
                        context -> {
                            JinferEmbeddingProperties p =
                                    context.getBean(JinferEmbeddingProperties.class);
                            assertThat(p.model()).isEqualTo("/emb.gguf");
                            assertThat(p.companions()).containsEntry("media", "/mmproj.gguf");
                            assertThat(p.contextCapacity()).isEqualTo(1024);
                            assertThat(p.dimensions()).isEqualTo(256);
                        });
    }

    @Test
    void theWidthDefaultsToTheModelsOwn() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.embedding.model=/emb.gguf")
                .run(
                        context ->
                                assertThat(
                                                context.getBean(JinferEmbeddingProperties.class)
                                                        .dimensions())
                                        .isZero()); // 0: the model's own width
    }

    @Test
    void thePublishedConstructorKeepsTheModelsOwnWidth() {
        JinferEmbeddingProperties p = new JinferEmbeddingProperties("/emb.gguf", 1024);
        assertThat(p.companions()).isNull();
        assertThat(p.contextCapacity()).isEqualTo(1024);
        assertThat(p.dimensions()).isZero();
    }

    @Test
    void aNegativeWidthIsRefusedBeforeTheModelLoads() {
        runner.withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=/missing.gguf",
                        "spring.ai.jinfer.embedding.dimensions=-1")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining(
                                            "dimensions must be 0 (the model's own width) or"
                                                    + " positive, not -1");
                        });
    }

    @Test
    void theContextCapacityDefaultSurvivesTheCompanions() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.embedding.model=/emb.gguf")
                .run(
                        context ->
                                assertThat(
                                                context.getBean(JinferEmbeddingProperties.class)
                                                        .contextCapacity())
                                        .isZero()); // 0: 4096, or less
    }

    @Test
    void aCompanionUrlIsRejectedBeforeResolution() {
        runner.withPropertyValues(
                        "spring.ai.model.embedding=jinfer",
                        "spring.ai.jinfer.embedding.model=/emb.gguf",
                        "spring.ai.jinfer.embedding.companions.media=https://example.org/mmproj.gguf")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining(
                                            "spring.ai.jinfer.embedding.companions.media is a URL;"
                                                    + " download it first");
                        });
    }

    @EnableConfigurationProperties(JinferEmbeddingProperties.class)
    static class PropsOnly {}

    @Configuration(proxyBeanMethods = false)
    static class UserEmbeddingConfiguration {
        @Bean
        EmbeddingModel userEmbeddingModel() {
            return mock(JinferEmbeddingModel.class);
        }
    }
}
