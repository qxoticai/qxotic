package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.media.Media;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.part.MediaPart;
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

    /**
     * A stretch of speech is a media chunk of its own under the response id, not stamped: the
     * aggregator carries no media, so a ChatClient sees the transcript and a raw subscriber
     * collects the audio from the chunks.
     */
    @Test
    void speechIsAMediaChunkTheAggregatorLeavesToTheSubscriber() {
        StreamingPartIndexer indexer = new StreamingPartIndexer();
        Media.Audio clip = new Media.Audio(new float[480], 24000, 1);
        ChatResponse spoken = JinferChatModel.chunk(MediaPart.of(JinferMappings.audio(clip)), "r1");
        assertEquals("r1", spoken.getMetadata().getId());
        AssistantMessage out = spoken.getResult().getOutput();
        assertEquals("", out.getText(), "a media chunk has no text");
        assertTrue(!StreamingParts.isPartial(out.getParts().get(0)), "media is never partial");
        assertEquals(JinferMappings.WAV, ((MediaPart) out.getParts().get(0)).media().getMimeType());

        AtomicReference<ChatResponse> aggregated = new AtomicReference<>();
        List<ChatResponse> passthrough =
                new MessageAggregator()
                        .aggregate(
                                Flux.just(
                                        JinferChatModel.chunk(
                                                indexer.stamp(TextPart.of("Paris")), "r1"),
                                        spoken,
                                        JinferChatModel.chunk(
                                                indexer.stamp(TextPart.of(".")), "r1")),
                                aggregated::set)
                        .collectList()
                        .block();
        assertEquals(3, passthrough.size(), "the subscriber sees every chunk, audio included");
        AssistantMessage whole = aggregated.get().getResult().getOutput();
        assertEquals("Paris.", whole.getText());
        assertEquals(List.<MessagePart>of(TextPart.of("Paris.")), whole.getParts());
    }
}
