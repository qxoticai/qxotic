package com.qxotic.jinfer.example.judgeadvisor;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.autoconfigure.JinferChatAutoConfiguration;
import com.qxotic.jinfer.testkit.ModelFixture;
import org.junit.jupiter.api.Assumptions;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.model.openai.autoconfigure.OpenAiChatAutoConfiguration;
import org.springframework.ai.model.tool.autoconfigure.ToolCallingAutoConfiguration;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * The hybrid loop as a Boot app would wire it: Kimi (OpenAI-compatible, key from env) generates,
 * the local GGUF judges. Gated on KIMI_API_KEY and the judge model file; skips otherwise. Run:
 * {@code KIMI_API_KEY=... mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-example-judge-advisor}
 */
@Tag("integration")
class JudgeAdvisorKimiIT {

    @Test
    void remoteGeniusLocalJudge() {
        String apiKey = System.getenv("KIMI_API_KEY");
        Assumptions.assumeTrue(apiKey != null && !apiKey.isBlank(), "KIMI_API_KEY not set");
        var judgeModel = ModelFixture.LFM25_8B_Q8.require();
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                JinferChatAutoConfiguration.class,
                                ToolCallingAutoConfiguration.class,
                                OpenAiChatAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + judgeModel,
                        "spring.ai.jinfer.chat.context-length=8192",
                        "spring.ai.jinfer.chat.max-tokens=512",
                        "spring.ai.openai.base-url=https://api.kimi.com/coding/v1",
                        "spring.ai.openai.api-key=" + apiKey,
                        "spring.ai.openai.chat.options.model=kimi-for-coding")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            String answer =
                                    JudgeAdvisorApplication.run(
                                            context.getBean(OpenAiChatModel.class),
                                            context.getBean(JinferChatModel.class));
                            assertThat(answer).isNotBlank();
                        });
    }
}
