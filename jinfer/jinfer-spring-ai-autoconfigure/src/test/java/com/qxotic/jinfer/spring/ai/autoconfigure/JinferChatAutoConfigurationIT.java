package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * Full wiring against a real GGUF: properties build the bean, the bean serves. Model-gated. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-spring-ai-autoconfigure}
 */
@Tag("integration")
class JinferChatAutoConfigurationIT {

    static final Path MODEL =
            Path.of(
                    System.getProperty(
                            "jinfer.testModel",
                            "/home/mukel/Desktop/playground/models/LiquidAI/LFM2.5-8B-A1B-Q8_0.gguf"));

    @Test
    void wiresAndServes() {
        Assumptions.assumeTrue(Files.exists(MODEL), "model not found: " + MODEL);
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferChatAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model-path=" + MODEL,
                        "spring.ai.jinfer.chat.context-length=4096",
                        "spring.ai.jinfer.chat.max-tokens=256")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(JinferChatModel.class);
                            String answer =
                                    context.getBean(JinferChatModel.class)
                                            .call(
                                                    "Answer with exactly one word: what is the"
                                                            + " capital of France?");
                            assertThat(answer).contains("Paris");
                        });
    }
}
