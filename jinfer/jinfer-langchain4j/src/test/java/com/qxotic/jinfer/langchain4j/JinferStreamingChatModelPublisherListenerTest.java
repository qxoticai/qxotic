package com.qxotic.jinfer.langchain4j;

import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.common.AbstractStreamingChatModelPublisherListenerTest;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterEach;

/**
 * langchain4j's listener contract for the reactive publisher, on the weightless model: a stream of
 * N chunks is N generated tokens, and a failed stream is a closed engine refusing the generation. A
 * dribbled stream needs no timing here - delivery is demand-driven, so a cancel or a throwing
 * {@code onNext} stops it before the complete response, however fast generation ran.
 */
class JinferStreamingChatModelPublisherListenerTest
        extends AbstractStreamingChatModelPublisherListenerTest {

    private final List<JinferChatModel> models = new ArrayList<>();

    @Override
    protected StreamingChatModel createModel(ChatModelListener listener, StreamScenario scenario) {
        JinferChatModel model =
                ChatFixtures.builder()
                        .contextCapacity(1024)
                        .maxOutputTokens(Math.max(1, scenario.contentChunks()))
                        .listeners(List.of(listener))
                        .build();
        models.add(model);
        if (scenario.fail()) model.close();
        return model.streaming();
    }

    @AfterEach
    void close() {
        models.forEach(JinferChatModel::close); // idempotent: a failed scenario's is closed already
    }
}
