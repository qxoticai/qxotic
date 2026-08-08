package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferDocumentPostProcessor;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/** Context behavior, no model needed: activation by model path, and property binding. */
class JinferRerankAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(AutoConfigurations.of(JinferRerankAutoConfiguration.class));

    @Test
    void dormantByDefault() {
        // an app that does not rerank configures nothing and loads no reranker weights
        runner.run(
                context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).doesNotHaveBean(JinferDocumentPostProcessor.class);
                });
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.rerank.model=/rerank.gguf",
                        "spring.ai.jinfer.rerank.context-length=1024",
                        "spring.ai.jinfer.rerank.instruction=Judge legal relevance",
                        "spring.ai.jinfer.rerank.top-k=3",
                        "spring.ai.jinfer.rerank.min-score=0.5")
                .run(
                        context -> {
                            JinferRerankProperties p =
                                    context.getBean(JinferRerankProperties.class);
                            assertThat(p.model()).isEqualTo("/rerank.gguf");
                            assertThat(p.contextLength()).isEqualTo(1024);
                            assertThat(p.instruction()).isEqualTo("Judge legal relevance");
                            assertThat(p.topK()).isEqualTo(3);
                            assertThat(p.minScore()).isEqualTo(0.5);
                        });
    }

    @Test
    void defaultsMatchTheAdapter() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.rerank.model=/rerank.gguf")
                .run(
                        context -> {
                            JinferRerankProperties p =
                                    context.getBean(JinferRerankProperties.class);
                            assertThat(p.contextLength()).isEqualTo(2048);
                            assertThat(p.instruction()).isNull(); // = the model card's wording
                            assertThat(p.topK()).isZero(); // keep every document
                            assertThat(p.minScore()).isZero(); // no relevance gate
                        });
    }

    @EnableConfigurationProperties(JinferRerankProperties.class)
    static class PropsOnly {}
}
