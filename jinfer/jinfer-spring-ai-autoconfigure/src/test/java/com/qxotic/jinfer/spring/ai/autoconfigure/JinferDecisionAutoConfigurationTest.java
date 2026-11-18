package com.qxotic.jinfer.spring.ai.autoconfigure;

import static org.assertj.core.api.Assertions.assertThat;

import com.qxotic.jinfer.spring.ai.JinferDecisionModel;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.springframework.ai.decision.DecisionModel;
import org.springframework.ai.decision.DecisionPrompt;
import org.springframework.ai.decision.DecisionResponse;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/** Context behavior, no model needed: selection, property binding, refusals, back-off. */
class JinferDecisionAutoConfigurationTest {

    private final ApplicationContextRunner runner =
            new ApplicationContextRunner()
                    .withConfiguration(
                            AutoConfigurations.of(JinferDecisionAutoConfiguration.class));

    @Test
    void dormantUnlessSelected() {
        // an app that asks no typed questions configures nothing and loads no weights
        runner.withPropertyValues("spring.ai.jinfer.decision.model=/d1.gguf")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).doesNotHaveBean(JinferDecisionModel.class);
                        });
        runner.withPropertyValues(
                        "spring.ai.model.decision=other",
                        "spring.ai.jinfer.decision.model=/d1.gguf")
                .run(context -> assertThat(context).doesNotHaveBean(JinferDecisionModel.class));
    }

    @Test
    void propertiesBind() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues(
                        "spring.ai.jinfer.decision.model=LiquidAI/d1-2B-GGUF:Q8_0",
                        "spring.ai.jinfer.decision.companions.media=/mmproj.gguf",
                        "spring.ai.jinfer.decision.context-capacity=2048",
                        "spring.ai.jinfer.decision.independent=true")
                .run(
                        context -> {
                            JinferDecisionProperties p =
                                    context.getBean(JinferDecisionProperties.class);
                            assertThat(p.model()).isEqualTo("LiquidAI/d1-2B-GGUF:Q8_0");
                            assertThat(p.companions()).isEqualTo(Map.of("media", "/mmproj.gguf"));
                            assertThat(p.contextCapacity()).isEqualTo(2048);
                            assertThat(p.independent()).isTrue();
                        });
    }

    @Test
    void defaultsMatchTheAdapter() {
        new ApplicationContextRunner()
                .withUserConfiguration(PropsOnly.class)
                .withPropertyValues("spring.ai.jinfer.decision.model=/d1.gguf")
                .run(
                        context -> {
                            JinferDecisionProperties p =
                                    context.getBean(JinferDecisionProperties.class);
                            assertThat(p.contextCapacity()).isZero(); // 0: 4096, or less
                            assertThat(p.independent()).isFalse(); // the model's own way
                            assertThat(p.companions()).isNull(); // text only
                        });
    }

    @Test
    void aMissingOrUrlModelIsRefusedNamingTheProperty() {
        runner.withPropertyValues("spring.ai.model.decision=jinfer")
                .run(
                        context ->
                                assertThat(context.getStartupFailure())
                                        .hasStackTraceContaining(
                                                "spring.ai.jinfer.decision.model is required"));
        runner.withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=https://example.org/d1.gguf")
                .run(
                        context ->
                                assertThat(context.getStartupFailure())
                                        .hasStackTraceContaining(
                                                "download the file and pass its local path"));
        runner.withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=/d1.gguf",
                        "spring.ai.jinfer.decision.companions.media=")
                .run(
                        context ->
                                assertThat(context.getStartupFailure())
                                        .hasStackTraceContaining(
                                                "spring.ai.jinfer.decision.companions.media must"
                                                        + " not be blank"));
    }

    @Test
    void anAppsOwnDecisionModelWins() {
        runner.withUserConfiguration(OwnModel.class)
                .withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=/never-loaded.gguf")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).doesNotHaveBean(JinferDecisionModel.class);
                            assertThat(context).hasSingleBean(DecisionModel.class);
                        });
    }

    /** A capacity the model cannot honor, and a companion that is a URL: refused at startup. */
    @Test
    void anImpossibleCapacityOrAUrlCompanionFailsAtStartup() {
        runner.withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=/d1.gguf",
                        "spring.ai.jinfer.decision.context-capacity=-5")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining(
                                            "contextCapacity must be 0 (the default), -1 (the"
                                                    + " model's maximum) or positive, got -5");
                        });
        runner.withPropertyValues(
                        "spring.ai.model.decision=jinfer",
                        "spring.ai.jinfer.decision.model=/d1.gguf",
                        "spring.ai.jinfer.decision.companions.media=https://example.org/m.gguf")
                .run(
                        context -> {
                            assertThat(context).hasFailed();
                            assertThat(context.getStartupFailure())
                                    .hasStackTraceContaining(
                                            "download the file and pass its local path");
                        });
    }

    /** A model that fails to load fails startup, in every context. */
    @Test
    void aModelThatFailsToLoadFailsStartup() {
        String d1 = "/nonexistent.gguf"; // the load fails, but the wiring is what this checks
        for (int i = 0; i < 2; i++)
            runner.withPropertyValues(
                            "spring.ai.model.decision=jinfer",
                            "spring.ai.jinfer.decision.model=" + d1)
                    .run(context -> assertThat(context).hasFailed());
    }

    @EnableConfigurationProperties(JinferDecisionProperties.class)
    static class PropsOnly {}

    @Configuration
    static class OwnModel {
        @Bean
        DecisionModel own() {
            return new DecisionModel() {
                @Override
                public DecisionResponse call(DecisionPrompt prompt) {
                    throw new UnsupportedOperationException();
                }
            };
        }
    }
}
