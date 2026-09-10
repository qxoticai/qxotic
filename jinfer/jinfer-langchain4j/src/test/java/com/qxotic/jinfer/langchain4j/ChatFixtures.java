package com.qxotic.jinfer.langchain4j;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import java.util.Optional;
import java.util.Set;

final class ChatFixtures {
    private ChatFixtures() {}

    static JinferChatModel.Builder builder() {
        var tokenizer = TestLanguageModel.TOKENIZER;
        ChatTemplate template =
                (conversation, capacity, sink) -> {
                    for (var message : conversation.messages()) {
                        int[] ids = tokenizer.encodeToArray(message.text());
                        if (ids.length > 0) sink.accept(Batch.prefill(ids));
                    }
                    sink.accept(Batch.step(0));
                    return new ChatTemplate.ReplyState(
                            IntSequence.empty(), ReplyParser.spans(tokenizer));
                };
        var loaded =
                new LoadedModel<>(
                        new TestLanguageModel(),
                        tokenizer,
                        "",
                        Set.of(),
                        new ContentKey("test-language-model"),
                        Optional.of(template),
                        LoadedModel.SamplingDefaults.NONE);
        return JinferChatModel.builder().model(loaded).temperature(0.0).maxOutputTokens(12);
    }
}
