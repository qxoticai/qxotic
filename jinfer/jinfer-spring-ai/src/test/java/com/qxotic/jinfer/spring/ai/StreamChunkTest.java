package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.part.MessagePart;
import org.springframework.ai.chat.messages.part.ReasoningPart;
import org.springframework.ai.chat.messages.part.StreamingPartIndexer;
import org.springframework.ai.chat.messages.part.StreamingParts;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.MessageAggregator;
import reactor.core.publisher.Flux;

/**
 * Streaming chunk shapes, no model needed: a reasoning delta is a {@link ReasoningPart} and a text
 * delta a {@link TextPart}, each stamped with its content block index, so {@link MessageAggregator}
 * (and ChatClient on top of it) rebuilds the reasoning as its own part instead of conflating it
 * with the content.
 */
class StreamChunkTest {

    @Test
    void aDeltaIsItsPartStampedPartial() {
        StreamingPartIndexer indexer = new StreamingPartIndexer();
        AssistantMessage thought =
                JinferChatModel.chunk(indexer.stamp(ReasoningPart.of("hmm")), "r1")
                        .getResult()
                        .getOutput();
        assertEquals("", thought.getText(), "a reasoning chunk has no text");
        assertEquals("hmm", JinferMappings.reasoning(thought));
        assertTrue(StreamingParts.isPartial(thought.getParts().get(0)));

        ChatResponse plain = JinferChatModel.chunk(indexer.stamp(TextPart.of("hi")), "r1");
        assertEquals("hi", plain.getResult().getOutput().getText());
        assertTrue(plain.getResult().getOutput().getReasoning().isEmpty());
        assertEquals("r1", plain.getMetadata().getId(), "every chunk of a response carries its id");
    }

    @Test
    void aggregatorKeepsReasoningOffContent() {
        StreamingPartIndexer indexer = new StreamingPartIndexer();
        AtomicReference<ChatResponse> aggregated = new AtomicReference<>();
        List<ChatResponse> passthrough =
                new MessageAggregator()
                        .aggregate(
                                Flux.just(
                                        JinferChatModel.chunk(
                                                indexer.stamp(ReasoningPart.of("let me think ")),
                                                "r1"),
                                        JinferChatModel.chunk(
                                                indexer.stamp(ReasoningPart.of("hard")), "r1"),
                                        JinferChatModel.chunk(
                                                indexer.stamp(TextPart.of("the answer ")), "r1"),
                                        JinferChatModel.chunk(
                                                indexer.stamp(TextPart.of("is 42")), "r1")),
                                aggregated::set)
                        .collectList()
                        .block();
        // the flux itself passes through untouched
        assertEquals(4, passthrough.size());
        // one part per run of deltas, in order, the streaming attributes gone
        AssistantMessage whole = aggregated.get().getResult().getOutput();
        assertEquals(
                List.<MessagePart>of(
                        ReasoningPart.of("let me think hard"), TextPart.of("the answer is 42")),
                whole.getParts());
        assertEquals("the answer is 42", whole.getText());
        assertEquals("let me think hard", JinferMappings.reasoning(whole));
    }
}
