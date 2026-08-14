package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.MessageAggregator;
import reactor.core.publisher.Flux;

/**
 * Streaming chunk shapes, no model needed: reasoning deltas carry core's {@code isThought} flag so
 * {@link MessageAggregator} (and ChatClient on top of it) accumulates them on the {@code thoughts}
 * lane instead of conflating them with content.
 */
class StreamChunkTest {

    @Test
    void thoughtChunksAreFlagged() {
        ChatResponse thought = JinferChatModel.chunk("hmm", true);
        assertEquals(
                true,
                thought.getResult().getOutput().getMetadata().get(JinferChatModel.IS_THOUGHT_KEY));

        ChatResponse plain = JinferChatModel.chunk("hi", false);
        assertNull(plain.getResult().getOutput().getMetadata().get(JinferChatModel.IS_THOUGHT_KEY));
    }

    @Test
    void aggregatorKeepsThoughtsOffContent() {
        AtomicReference<ChatResponse> aggregated = new AtomicReference<>();
        List<ChatResponse> passthrough =
                new MessageAggregator()
                        .aggregate(
                                Flux.just(
                                        JinferChatModel.chunk("let me think ", true),
                                        JinferChatModel.chunk("hard", true),
                                        JinferChatModel.chunk("the answer ", false),
                                        JinferChatModel.chunk("is 42", false)),
                                aggregated::set)
                        .collectList()
                        .block();
        // the flux itself passes through untouched
        assertEquals(4, passthrough.size());
        // core's convention: flagged chunks accumulate on the thoughts lane, not just content
        assertEquals(
                "let me think hard",
                aggregated.get().getResult().getOutput().getMetadata().get("thoughts"));
    }
}
