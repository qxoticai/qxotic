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
 * high-level API) against JinferChatModel on Inkling 2 Mini - the shape applications actually use,
 * over the same provider {@code Inkling2ToolIT} drives at the wire level.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.Inkling2AiServiceIT#modelAvailable")
class Inkling2AiServiceIT extends AbstractAiServiceIT {

    static final String REF = "hf.co/ggml-org/Inkling-2-Mini-GGUF/Inkling-2-Mini-Q8_0.gguf";

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    private static JinferChatModel model;

    @AfterAll
    void unload() {
        if (model != null) model.close();
    }

    static JinferChatModel shared() {
        if (model == null) {
            model =
                    JinferChatModel.builder()
                            .modelPath(TestModels.require(REF))
                            .contextCapacity(8192)
                            .maxOutputTokens(512)
                            .temperature(0.0)
                            .thinking(true)
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
