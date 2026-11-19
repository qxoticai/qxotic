package com.qxotic.jinfer.spring.ai.starter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.autoconfigure.JinferChatAutoConfiguration;
import com.qxotic.jinfer.spring.ai.autoconfigure.JinferSessionAutoConfiguration;
import com.qxotic.jinfer.testkit.TestModels;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistryAssert;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.model.chat.client.autoconfigure.ChatClientAutoConfiguration;
import org.springframework.ai.model.chat.observation.autoconfigure.ChatObservationAutoConfiguration;
import org.springframework.ai.model.session.autoconfigure.SessionServiceAutoConfiguration;
import org.springframework.ai.model.tool.autoconfigure.ToolCallingAutoConfiguration;
import org.springframework.ai.session.SessionService;
import org.springframework.ai.session.advisor.SessionAdvisors;
import org.springframework.ai.support.ToolCallbacks;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.boot.autoconfigure.AutoConfigurations;
import org.springframework.boot.micrometer.metrics.autoconfigure.CompositeMeterRegistryAutoConfiguration;
import org.springframework.boot.micrometer.metrics.autoconfigure.MetricsAutoConfiguration;
import org.springframework.boot.micrometer.metrics.autoconfigure.export.simple.SimpleMetricsExportAutoConfiguration;
import org.springframework.boot.micrometer.observation.autoconfigure.ObservationAutoConfiguration;
import org.springframework.boot.test.context.runner.ApplicationContextRunner;

/**
 * Dogfoods the starter's dependency graph the way a Boot app would: the auto-configured
 * JinferChatModel plus the ChatClient stack (ToolCallingAdvisor auto-registered on the builder
 * bean), running a full tool loop against a real GGUF. Model-gated via {@link TestModels}. Run:
 * {@code mvn test -Dsurefire.excludedGroups= -Dgroups=integration -pl
 * jinfer-spring-ai-spring-boot-starter}
 */
@Tag("integration")
class JinferStarterIT {

    static Path MODEL; // resolved in @BeforeAll: a missing model then skips

    @BeforeAll
    static void requireModels() {
        MODEL = TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf");
    }

    static class Weather {
        @Tool(description = "Get current weather for a city")
        String weather(String city) {
            return "18C, sunny in " + city;
        }
    }

    @Test
    void chatClientToolLoopEndToEnd() {
        ToolCallback weather = ToolCallbacks.from(new Weather())[0];
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                JinferChatAutoConfiguration.class,
                                ToolCallingAutoConfiguration.class,
                                ChatClientAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-capacity=4096",
                        "spring.ai.jinfer.chat.max-tokens=512")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(JinferChatModel.class);
                            assertThat(context).hasSingleBean(ChatClient.Builder.class);
                            // the advisor loop: model proposes the call, framework executes it,
                            // model grounds the answer in the tool result
                            String answer =
                                    context.getBean(ChatClient.Builder.class)
                                            .build()
                                            .prompt("What is the weather in Paris?")
                                            .tools(weather)
                                            .call()
                                            .content();
                            assertThat(answer).contains("18");
                        });
    }

    @Test
    void sessionMemoryRemembersAcrossTurns() {
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                JinferChatAutoConfiguration.class,
                                ChatClientAutoConfiguration.class,
                                JinferSessionAutoConfiguration.class,
                                SessionServiceAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-capacity=4096",
                        "spring.ai.jinfer.chat.max-tokens=64",
                        "spring.ai.jinfer.chat.temperature=0",
                        "spring.ai.jinfer.chat.seed=1")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            assertThat(context).hasSingleBean(SessionService.class);
                            ChatClient chat =
                                    context.getBean(ChatClient.Builder.class)
                                            .defaultAdvisors(
                                                    SessionAdvisors.builder(
                                                                    context.getBean(
                                                                            SessionService.class))
                                                            .build())
                                            .build();
                            chat.prompt("My name is Zorblax.")
                                    .advisors(SessionAdvisors.session("s1"))
                                    .call()
                                    .content();
                            String answer =
                                    chat.prompt("What is my name?")
                                            .advisors(SessionAdvisors.session("s1"))
                                            .call()
                                            .content();
                            assertThat(answer).containsIgnoringCase("Zorblax");
                            // another session knows nothing of it
                            String other =
                                    chat.prompt("What is my name?")
                                            .advisors(SessionAdvisors.session("s2"))
                                            .call()
                                            .content();
                            assertThat(other).doesNotContainIgnoringCase("Zorblax");
                            // the turns went into the session, two of them against one
                            SessionService sessions = context.getBean(SessionService.class);
                            assertThat(sessions.getEventVersion("s1"))
                                    .isGreaterThan(sessions.getEventVersion("s2"));
                            assertThat(sessions.getEventVersion("s2")).isGreaterThan(0);
                        });
    }

    /** The starter records the gen_ai metrics once the application has a meter registry. */
    @Test
    void aChatRecordsTheGenAiMetrics() {
        new ApplicationContextRunner()
                .withConfiguration(
                        AutoConfigurations.of(
                                ObservationAutoConfiguration.class,
                                MetricsAutoConfiguration.class,
                                CompositeMeterRegistryAutoConfiguration.class,
                                SimpleMetricsExportAutoConfiguration.class,
                                ChatObservationAutoConfiguration.class,
                                JinferChatAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-capacity=4096",
                        "spring.ai.jinfer.chat.max-tokens=32")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            context.getBean(JinferChatModel.class).call("One word: ok?");
                            MeterRegistry meters = context.getBean(MeterRegistry.class);
                            assertThat(
                                            meters.find("gen_ai.client.token.usage")
                                                    .tag("gen_ai.token.type", "input")
                                                    .tag("gen_ai.system", "jinfer")
                                                    .counter())
                                    .isNotNull()
                                    .satisfies(c -> assertThat(c.count()).isPositive());
                        });
    }

    @Test
    void observationRegistryIsWiredFromTheContext() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferChatAutoConfiguration.class))
                .withBean(TestObservationRegistry.class, TestObservationRegistry::create)
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-capacity=4096",
                        "spring.ai.jinfer.chat.max-tokens=32")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            context.getBean(JinferChatModel.class).call("One word: ok?");
                            TestObservationRegistryAssert.assertThat(
                                            context.getBean(TestObservationRegistry.class))
                                    .hasNumberOfObservationsWithNameEqualTo(
                                            "gen_ai.client.operation", 1);
                            TestObservationRegistryAssert.assertThat(
                                            context.getBean(TestObservationRegistry.class))
                                    .hasAnObservationWithAKeyValue("gen_ai.system", "jinfer");
                        });
    }

    @Test
    void contextShutdownClosesTheModel() {
        var bean = new AtomicReference<JinferChatModel>();
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferChatAutoConfiguration.class))
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-capacity=4096",
                        "spring.ai.jinfer.chat.max-tokens=32")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            bean.set(context.getBean(JinferChatModel.class));
                        });
        // Boot inferred close() as the destroy method (AutoCloseable)
        assertThatThrownBy(() -> bean.get().call("hi"))
                .isInstanceOf(IllegalStateException.class)
                .hasMessageContaining("closed");
    }
}
