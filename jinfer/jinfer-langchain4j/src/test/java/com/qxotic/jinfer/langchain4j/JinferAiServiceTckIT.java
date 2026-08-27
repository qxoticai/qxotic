package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.common.AbstractAiServiceIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j AiServices compliance kit (chat + tools + structured-output interplay through the
 * high-level API) against JinferChatModel on LFM2.5-8B. The kit drives the same provider the
 * low-level TCK certifies, but through {@code AiServices.create(...)} - the shape applications
 * actually use.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferAiServiceTckIT#modelAvailable")
class JinferAiServiceTckIT extends AbstractAiServiceIT {

    static boolean modelAvailable() {
        return TestModels.find(JinferChatModelTckIT.REF).isPresent();
    }

    private static JinferChatModel model;

    @AfterAll
    static void unload() {
        if (model != null) model.close();
    }

    static JinferChatModel shared() {
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(TestModels.require(JinferChatModelTckIT.REF))
                            .contextLength(8192)
                            .maxOutputTokens(512)
                            // pinned greedy, same reason as the low-level TCK: the kit tests the
                            // CONTRACT, not sampling quality, and must not flake
                            .temperature(0.0)
                            .thinking(JinferChatModelTckIT.tckThinking())
                            .seed(7L)
                            .build();
        }
        return model;
    }

    @Override
    protected List<ChatModel> models() {
        return List.of(TckShield.chat(shared()));
    }

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(ChatModel chatModel) {
        return JinferTokenUsage.class;
    }
}
