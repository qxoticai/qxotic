package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.output.TokenUsage;
import dev.langchain4j.service.common.AbstractStreamingAiServiceIT;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j streaming AiServices kit ({@code TokenStream}: partial-response ordering,
 * completion metadata, tool execution mid-stream) against JinferStreamingChatModel on LFM2.5-8B -
 * the streaming twin of {@link JinferAiServiceTckIT}, sharing one engine via {@code
 * JinferChatModel.streaming()}.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferStreamingAiServiceTckIT#modelAvailable")
class JinferStreamingAiServiceTckIT extends AbstractStreamingAiServiceIT {

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
    protected List<StreamingChatModel> models() {
        return List.of(TckShield.streaming(shared().streaming()));
    }

    @Override
    protected Class<? extends TokenUsage> tokenUsageType(StreamingChatModel streamingChatModel) {
        return JinferTokenUsage.class;
    }
}
