package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferTranscriptionModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/** Context behavior, no model needed: activation by model path, and property binding. */
class JinferTranscriptionAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(
                            AutoConfigurations.of(JinferTranscriptionAutoConfiguration.class));

    @Test
    void dormantByDefault() {
        // an app that does not transcribe configures nothing and loads no weights
        runner.run(
                context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).doesNotHaveBean(JinferTranscriptionModel.class);
                });
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.transcription.model=/parakeet.gguf")
                .run(
                        context -> {
                            JinferTranscriptionProperties p =
                                    context.getBean(JinferTranscriptionProperties.class);
                            assertThat(p.model()).isEqualTo("/parakeet.gguf");
                        });
    }

    @Test
    void blankModelFailsBeforeFilesystemAccess() {
        for (String model : new String[] {"", "   "}) {
            runner.withPropertyValues("spring.ai.jinfer.transcription.model=" + model)
                    .run(
                            context -> {
                                assertThat(context).hasFailed();
                                assertThat(context.getStartupFailure())
                                        .hasRootCauseInstanceOf(IllegalStateException.class)
                                        .hasStackTraceContaining(
                                                "spring.ai.jinfer.transcription.model")
                                        .hasStackTraceContaining("is required")
                                        .hasStackTraceContaining("local path")
                                        .hasStackTraceContaining("model ref");
                            });
        }
    }

    @Test
    void modelUrlIsRejectedBeforeResolution() {
        runner.withPropertyValues(
                        "spring.ai.jinfer.transcription.model=https://example.org/model.gguf")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining("download it first");
                        });
    }

    /** Registered, or a Boot app would never see the bean however it is configured. */
    @Test
    void isRegisteredForAutoConfigurationImport() throws Exception {
        String imports =
                new String(
                        JinferTranscriptionAutoConfigurationTest.class
                                .getClassLoader()
                                .getResourceAsStream(
                                        "META-INF/spring/"
                                            + "org.springframework.boot.autoconfigure.AutoConfiguration.imports")
                                .readAllBytes());
        assertThat(imports).contains(JinferTranscriptionAutoConfiguration.class.getName());
    }

    @EnableConfigurationProperties(JinferTranscriptionProperties.class)
    static class PropsOnly {}
}
