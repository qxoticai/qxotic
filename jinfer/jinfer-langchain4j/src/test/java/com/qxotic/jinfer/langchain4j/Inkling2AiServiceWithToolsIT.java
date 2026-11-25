package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.service.common.AbstractAiServiceWithToolsIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j tools battery through AiServices ({@code AbstractAiServiceWithToolsIT}: POJO
 * parameters, nested POJOs, enums, lists of strings/integers/POJOs, map parameters) against
 * JinferChatModel on Inkling 2 Mini.
 *
 * <p>The kit's {@code @Tool} methods carry no descriptions, so the model sees {@code add(arg0,
 * arg1)}; deciding that such a declaration applies is a capability, and this checkpoint's tool
 * scores come from {@code Inkling2ToolIT}. This suite certifies ARGUMENT TYPING on the high-level
 * API - which is a different question, and the one applications feel.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.Inkling2AiServiceWithToolsIT#modelAvailable")
class Inkling2AiServiceWithToolsIT extends AbstractAiServiceWithToolsIT {

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
                            // thinking ON: tool calling on this family is gated on the reasoning
                            // lane (measured in Inkling2ToolCallingProbe's lane matrix)
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
}
