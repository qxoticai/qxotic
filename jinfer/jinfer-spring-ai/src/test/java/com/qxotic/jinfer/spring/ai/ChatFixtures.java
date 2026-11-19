package com.qxotic.jinfer.spring.ai;

import com.qxotic.jinfer.Batch;
import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.ChatTemplate;
import com.qxotic.jinfer.chat.LoadedModel;
import com.qxotic.jinfer.chat.ReplyParser;
import com.qxotic.jinfer.testkit.TestLanguageModel;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.MessageAggregator;
import reactor.core.publisher.Flux;

/**
 * A weightless chat model - the testkit's language model behind a one-line template - and helpers.
 */
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
        return JinferChatModel.builder()
                .model(loaded)
                .options(JinferChatOptions.builder().temperature(0.0).maxTokens(12).build());
    }

    /** A stream as ChatClient aggregates it. */
    static AssistantMessage aggregate(List<ChatResponse> chunks) {
        AtomicReference<ChatResponse> whole = new AtomicReference<>();
        new MessageAggregator().aggregate(Flux.fromIterable(chunks), whole::set).blockLast();
        return whole.get().getResult().getOutput();
    }
}
