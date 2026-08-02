package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferSpeechModel;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/** Context behavior, no model needed: activation by model path, and property binding. */
class JinferSpeechAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(AutoConfigurations.of(JinferSpeechAutoConfiguration.class));

    @Test
    void dormantByDefault() {
        // an app that does not speak configures nothing and loads no speech weights
        runner.run(
                context -> {
                    assertThat(context).hasNotFailed();
                    assertThat(context).doesNotHaveBean(JinferSpeechModel.class);
                });
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.speech.model-path=/speech.gguf",
                        "spring.ai.jinfer.speech.speed=1.25",
                        "spring.ai.jinfer.speech.max-input-chars=500")
                .run(
                        context -> {
                            JinferSpeechProperties p =
                                    context.getBean(JinferSpeechProperties.class);
                            assertThat(p.modelPath()).isEqualTo("/speech.gguf");
                            assertThat(p.speed()).isEqualTo(1.25);
                            assertThat(p.maxInputChars()).isEqualTo(500);
                        });
    }

    @Test
    void defaultsLeaveTheAdaptersOwnChoices() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.speech.model-path=/speech.gguf")
                .run(
                        context -> {
                            JinferSpeechProperties p =
                                    context.getBean(JinferSpeechProperties.class);
                            // 0 is "unset", not a value: the autoconfiguration must not pass these
                            // through and override the port's own pace and input bound
                            assertThat(p.speed()).isZero();
                            assertThat(p.maxInputChars()).isZero();
                        });
    }

    /** Registered, or a Boot app would never see the bean however it is configured. */
    @Test
    void isRegisteredForAutoConfigurationImport() throws Exception {
        String imports =
                new String(
                        JinferSpeechAutoConfigurationTest.class
                                .getClassLoader()
                                .getResourceAsStream(
                                        "META-INF/spring/"
                                            + "org.springframework.boot.autoconfigure.AutoConfiguration.imports")
                                .readAllBytes());
        assertThat(imports).contains(JinferSpeechAutoConfiguration.class.getName());
    }

    @EnableConfigurationProperties(JinferSpeechProperties.class)
    static class PropsOnly {}
}
