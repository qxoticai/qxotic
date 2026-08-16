package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.testkit.TestModels;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.common.AbstractChatModelListenerIT;
import dev.langchain4j.model.chat.listener.ChatModelListener;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The langchain4j listener compliance kit against {@link JinferChatModel} on LFM2.5-8B: the
 * listener must observe the exact request, the response metadata, and - on a failing model - the
 * error with the request context intact. jinfer's failing model is a CLOSED one: use-after-close is
 * an {@link IllegalStateException}, and core's {@code ChatModel.chat} wrapper reports it to
 * listeners before rethrowing. Model-gated via @EnabledIf.
 */
@Tag("integration")
@EnabledIf("com.qxotic.jinfer.langchain4j.JinferChatModelListenerIT#modelAvailable")
class JinferChatModelListenerIT extends AbstractChatModelListenerIT {

    static final String REF = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF/LFM2.5-8B-A1B-Q8_0.gguf";

    static boolean modelAvailable() {
        return TestModels.find(REF).isPresent();
    }

    // createModel/createFailingModel products are built inside kit test bodies - nothing closes
    // them unless we track them (same reason as the TCK's `created` list)
    private static final List<JinferChatModel> created =
            Collections.synchronizedList(new ArrayList<>());

    @AfterAll
    static void unload() {
        created.forEach(JinferChatModel::close);
        created.clear();
    }

    private String name;

    @Override
    protected ChatModel createModel(ChatModelListener listener) {
        JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(4096)
                        // the kit's own defaults, set explicitly so defaultRequestParameters()
                        // echoes them: thinking OFF because the kit's 7-token budget must buy
                        // answer tokens, not analysis
                        .temperature(0.7)
                        .topP(1.0)
                        .maxOutputTokens(7)
                        .thinking(false)
                        .listeners(List.of(listener))
                        .build();
        name = m.defaultRequestParameters().modelName();
        created.add(m);
        return m;
    }

    @Override
    protected String modelName() {
        return name; // captured from the live engine: the kit compares against what WE report
    }

    @Override
    protected ChatModel createFailingModel(ChatModelListener listener) {
        JinferChatModel m =
                JinferChatModel.builder()
                        .modelPath(TestModels.require(REF))
                        .contextLength(512)
                        .listeners(List.of(listener))
                        .build();
        m.close(); // use-after-close is jinfer's honest call-time failure (a missing GGUF fails
        // at BUILD time and could never be returned here)
        return m;
    }

    @Override
    protected Class<? extends Exception> expectedExceptionClass() {
        return IllegalStateException.class;
    }
}
