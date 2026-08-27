package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.testkit.TestModels;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIf;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.test.chat.client.advisor.AbstractToolCallingAdvisorAutoRegistrationIT;

/**
 * The auto-registration twin of {@link JinferToolCallingAdvisorIT}: Spring AI registers the
 * ToolCallingAdvisor implicitly when a request carries tools, so the same loops must hold with no
 * explicit advisor - including the advisor-chain iteration counts and user-controlled tool
 * execution. Certifies that jinfer's provider does not depend on the explicit-advisor path.
 */
@Tag("integration")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
@EnabledIf("com.qxotic.jinfer.spring.ai.JinferToolCallingAdvisorAutoRegistrationIT#modelAvailable")
class JinferToolCallingAdvisorAutoRegistrationIT
        extends AbstractToolCallingAdvisorAutoRegistrationIT {

    static boolean modelAvailable() {
        return TestModels.find(JinferToolCallingAdvisorIT.REF).isPresent();
    }

    private static JinferChatModel model;

    @BeforeAll
    static void load() {
        model =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(JinferToolCallingAdvisorIT.REF))
                        .contextLength(8192)
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
