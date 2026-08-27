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
 * parameters, nested POJOs, enums, lists of strings/integers/POJOs, map parameters...) against
 * JinferChatModel on LFM2.5-8B. Complements the family tool batteries ({@code Lfm2ToolIT} and
 * friends), which pin wire fidelity per chat template; this suite certifies the ARGUMENT TYPING
 * contract the high-level API promises.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferAiServiceWithToolsTckIT#modelAvailable")
class JinferAiServiceWithToolsTckIT extends AbstractAiServiceWithToolsIT {

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
}
