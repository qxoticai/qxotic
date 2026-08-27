package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.TestModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.test.chat.client.advisor.AbstractToolCallingAdvisorIT;

/**
 * Spring AI's own reusable battery ({@code spring-ai-test}'s {@link AbstractToolCallingAdvisorIT})
 * against JinferChatModel on LFM2.5-8B: the ChatClient + ToolCallingAdvisor loop, blocking and
 * streaming, with external memory and returnDirect. It drives the provider through the advisor
 * chain - the shape Spring applications actually use - where {@code AbstractCapabilityIT} pins the
 * raw mapping layer.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@EnabledIf("com.qxotic.jinfer.spring.ai.JinferToolCallingAdvisorIT#modelAvailable")
class JinferToolCallingAdvisorIT extends AbstractToolCallingAdvisorIT {

    static final String REF = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    private static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(8192)
                        // the advisor loop re-sends the whole exchange per round; greedy keeps a
                        // multi-round tool battery from flaking on near-ties
                        .options(
                                JinferChatOptions.builder().maxTokens(512).temperature(0.0).build())
                        .build();
    }

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    @Override
    protected ChatModel getChatModel() {
        return model;
    }
}
