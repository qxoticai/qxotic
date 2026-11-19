package com.qxotic.jinfer.langchain4j;

import static org.junit.jupiter.api.Assertions.assertEquals;

import com.qxotic.jinfer.ContentKey;
import com.qxotic.jinfer.chat.Verbatim;
import com.qxotic.toknroll.IntSequence;
import dev.langchain4j.data.message.AiMessage;
import org.junit.jupiter.api.Test;

/** The table of recent replies: keyed by what a reply says, bounded by the token ids it holds. */
class RepliesTest {

    private static Verbatim record(int... positions) {
        return new Verbatim(
                new ContentKey("test-model"), IntSequence.of(positions), IntSequence.empty());
    }

    @Test
    void theLeastRecentlyUsedGoesFirstOnceTheBudgetIsFull() {
        Replies replies = new Replies(4); // token ids
        replies.remember(AiMessage.from("a"), record(1, 2));
        replies.remember(AiMessage.from("b"), record(3, 4));
        replies.recall(AiMessage.from("a")); // a is now the most recently used
        replies.remember(AiMessage.from("c"), record(5));
        assertEquals(Verbatim.NONE, replies.recall(AiMessage.from("b")), "evicted");
        assertEquals(record(1, 2), replies.recall(AiMessage.from("a")));
        assertEquals(record(5), replies.recall(AiMessage.from("c")));
    }

    @Test
    void aRecordOverTheWholeBudgetIsNotKept() {
        Replies replies = new Replies(2);
        replies.remember(AiMessage.from("long"), record(1, 2, 3));
        assertEquals(Verbatim.NONE, replies.recall(AiMessage.from("long")));
    }
}
