package com.qxotic.jinfer.spring.ai;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Verbatim;
import com.qxotic.toknroll.IntSequence;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.part.ReasoningPart;
import org.springframework.ai.chat.messages.part.TextPart;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;

/**
 * The table of recent replies: keyed by what a reply says, bounded by the token ids it holds, and
 * filled with exactly what the model delivers.
 */
class RepliesTest {

    private static Verbatim record(int... positions) {
        return new Verbatim(
                new ContentKey("test-model"), IntSequence.of(positions), IntSequence.empty());
    }

    private static AssistantMessage reply(String reasoning, String text) {
        return AssistantMessage.builder()
                .part(ReasoningPart.of(reasoning))
                .part(TextPart.of(text))
                .build();
    }

    /** A streamed echo lacks the reasoning span's seed newline, which never streamed. */
    @Test
    void theReasoningCountsStripped() {
        Replies replies = new Replies();
        Verbatim record = record(1, 2);
        replies.remember(reply("\nchecking", "Sunny."), record);
        assertEquals(record, replies.recall(reply("checking", "Sunny.")));
        assertEquals(Verbatim.NONE, replies.recall(reply("checking", "Rainy.")));
    }

    /** What the caller gets is what the model remembers; a stop-sequence cut is not. */
    @Test
    void aCallRemembersTheReplyItDelivers() {
        try (JinferChatModel model = ChatFixtures.builder().build()) {
            AssistantMessage whole = model.call(new Prompt("hello")).getResult().getOutput();
            assertNotEquals(Verbatim.NONE, model.replies.recall(whole));

            String text = whole.getText();
            String stop = text.substring(text.length() / 2, text.length() / 2 + 1);
            AssistantMessage cut =
                    model.call(
                                    new Prompt(
                                            "hello",
                                            JinferChatOptions.builder()
                                                    .stopSequences(List.of(stop))
                                                    .build()))
                            .getResult()
                            .getOutput();
            assertTrue(cut.getText().length() < text.length(), "the stop sequence cut nothing");
            assertEquals(Verbatim.NONE, model.replies.recall(cut));
        }
    }

    /** A stream's chunks, aggregated as ChatClient does, add up to a reply the model remembers. */
    @Test
    void aStreamRemembersTheReplyItsChunksAddUpTo() {
        try (JinferChatModel model = ChatFixtures.builder().build()) {
            List<ChatResponse> chunks = model.stream(new Prompt("hello")).collectList().block();
            assertNotEquals(Verbatim.NONE, model.replies.recall(ChatFixtures.aggregate(chunks)));
        }
    }

    @Test
    void theLeastRecentlyUsedGoesFirstOnceTheBudgetIsFull() {
        Replies replies = new Replies(4); // token ids
        replies.remember(reply("", "a"), record(1, 2));
        replies.remember(reply("", "b"), record(3, 4));
        replies.recall(reply("", "a")); // a is now the most recently used
        replies.remember(reply("", "c"), record(5));
        assertEquals(Verbatim.NONE, replies.recall(reply("", "b")), "evicted");
        assertEquals(record(1, 2), replies.recall(reply("", "a")));
    }
}
