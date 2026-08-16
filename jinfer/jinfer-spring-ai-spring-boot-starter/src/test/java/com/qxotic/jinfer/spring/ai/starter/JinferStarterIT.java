package com.qxotic.jinfer.spring.ai.starter;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.qxotic.jinfer.spring.ai.JinferChatModel;
import com.qxotic.jinfer.spring.ai.autoconfigure.JinferChatAutoConfiguration;
import com.qxotic.jinfer.testkit.TestModels;
import io.micrometer.observation.tck.TestObservationRegistry;
import java.nio.file.Path;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.model.chat.client.autoconfigure.ChatClientAutoConfiguration;
import org.springframework.ai.model.tool.autoconfigure.ToolCallingAutoConfiguration;
import org.springframework.ai.support.ToolCallbacks;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.annotation.Tool;
import org.springframework.boot.autoconfigure.AutoConfigurations;
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

    static final Path MODEL =
            TestModels.require("hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf");

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
                        "spring.ai.jinfer.chat.context-length=4096",
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
                                            .toolCallbacks(weather)
                                            .call()
                                            .content();
                            assertThat(answer).contains("18");
                        });
    }

    @Test
    void observationRegistryIsWiredFromTheContext() {
        new ApplicationContextRunner()
                .withConfiguration(AutoConfigurations.of(JinferChatAutoConfiguration.class))
                .withBean(TestObservationRegistry.class, TestObservationRegistry::create)
                .withPropertyValues(
                        "spring.ai.jinfer.chat.model=" + MODEL,
                        "spring.ai.jinfer.chat.context-length=4096",
                        "spring.ai.jinfer.chat.max-tokens=32")
                .run(
                        context -> {
                            assertThat(context).hasNotFailed();
                            context.getBean(JinferChatModel.class).call("One word: ok?");
                            context.getBean(TestObservationRegistry.class)
                                    .assertThat()
                                    .hasNumberOfObservationsWithNameEqualTo(
                                            "gen_ai.client.operation", 1);
                            context.getBean(TestObservationRegistry.class)
                                    .assertThat()
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
                        "spring.ai.jinfer.chat.context-length=4096",
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
