package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.qxotic.jinfer.spring.ai.JinferDecisionModel;
import com.qxotic.jinfer.testkit.TestModels;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.ai.decision.question.YesNoQuestion;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * A Boot app selects jinfer for decisions and gets Liquid d1-2B, its vision companion attached,
 * from properties alone. Model-gated via {@link TestModels}.
 */
@Tag("integration")
class JinferDecisionAutoConfigurationIT {

    /**
     * The knobs are behaviour, not just binding: independent asks each question alone, companions
     * attach, the bean injects.
     */
    @Test
    void theConfiguredKnobsReachTheModel() {
        String d1 = TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf").toString();
        String mmproj =
                TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf").toString();
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferDecisionAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=" + d1,
                        "spring.ai.jinfer.decision.companions.media=" + mmproj,
                        "spring.ai.jinfer.decision.context-capacity=2048",
                        "spring.ai.jinfer.decision.independent=true")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            // the bean is injectable as the framework's interface
                            DecisionModel model = context.getBean(DecisionModel.class);
                            assertThat(model).isInstanceOf(JinferDecisionModel.class);
                            // each question read alone: both answer
                            DecisionResponse response =
                                    model.call(
                                            "I was charged twice and I want my money back today.",
                                            Map.of(
                                                    "refund",
                                                    YesNoQuestion.of(
                                                            "Is the customer asking for a"
                                                                    + " refund?"),
                                                    "urgent",
                                                    YesNoQuestion.of("Is this urgent?")));
                            assertThat(response.getYesNo("refund").isYes()).isTrue();
                            assertThat(response.hasAnswer("urgent")).isTrue();
                        });
    }

    /** Spring closes the bean with the context: the weights and state are freed, not leaked. */
    @Test
    void closingTheContextClosesTheModel() {
        String d1 = TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf").toString();
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferDecisionAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.decision=jinfer", "spring.ai.jinfer.decision.model=" + d1)
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            JinferDecisionModel model = context.getBean(JinferDecisionModel.class);
                            assertThat(model.call("x", Map.of("q", YesNoQuestion.of("?"))))
                                    .isNotNull();
                            context.close(); // Spring's inferred destroy method is close()
                            assertThatThrownBy(
                                            () ->
                                                    model.call(
                                                            "x",
                                                            Map.of("q", YesNoQuestion.of("?"))))
                                    .isInstanceOf(IllegalStateException.class);
                        });
    }

    @Test
    void aBootAppAsksD1() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferDecisionAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model="
                                + TestModels.require("hf.co/LiquidAI/d1-2B-GGUF/d1-2B-Q8_0.gguf"),
                        "spring.ai.jinfer.decision.companions.media="
                                + TestModels.require(
                                        "hf.co/LiquidAI/d1-2B-GGUF/mmproj-d1-2B-F16.gguf"),
                        "spring.ai.jinfer.decision.context-capacity=1024")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            DecisionResponse response =
                                    context.getBean(DecisionModel.class)
                                            .call(
                                                    "I was charged twice for my order and I want"
                                                            + " my money back today.",
                                                    Map.of(
                                                            "refund",
                                                            YesNoQuestion.of(
                                                                    "Is the customer asking for a"
                                                                            + " refund?")));
                            assertThat(response.getYesNo("refund").isYes()).isTrue();
                        });
    }
}
