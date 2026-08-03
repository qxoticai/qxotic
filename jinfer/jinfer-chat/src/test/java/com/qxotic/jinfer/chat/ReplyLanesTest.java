package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.Batch;
import java.util.List;
import java.util.Optional;
import org.junit.jupiter.api.Test;

/**
 * Whether a reply's call syntax is CLAIMED as a tool call, or left as visible text.
 *
 * <p>The engine used to always claim it. The server deliberately did not when the caller offered no
 * tools, and the server was right: a claimed call the client never asked for is not a call, it is
 * an answer the client cannot see. LFM2.5 answers a bare prompt with its own final_output call,
 * which turned a plain reply into a tool-call response carrying no text at all.
 *
 * <p>Pinned here, over the fake vocabulary, because the only thing covering it was the server's
 * integration battery - which is opt-in and does not run in a normal build.
 */
final class ReplyLanesTest {

    private static final int THINK_OPEN = 0;
    private static final int THINK_CLOSE = 1;
    private static final int CALL_OPEN = 2;
    private static final int CALL_CLOSE = 3;
    private static final int HELLO = 5;
    private static final int PAYLOAD = 7; // "[f(x=1)]"

    /** A template whose parser claims calls between the fake vocabulary's call markers. */
    private static final ChatTemplate CLAIMING =
            new ChatTemplate() {
                @Override
                public List<Batch> encode(Conversation conversation) {
                    throw new UnsupportedOperationException("not used");
                }

                @Override
                public ReplyParser parser() {
                    return ReplyParser.spans(
                            ReplyParserTest.TOK,
                            "<|call|>",
                            "<|/call|>",
                            ToolCallSyntax::parseBlock);
                }
            };

    private static Message feed(boolean claimToolCalls, int... tokens) {
        ReplyLanes lanes =
                new ReplyLanes(
                        Optional.of(CLAIMING), ReplyParserTest.TOK, new int[0], claimToolCalls);
        StringBuilder shown = new StringBuilder();
        for (int token : tokens) shown.append(lanes.feed(token));
        Message reply = lanes.finish();
        // what the caller SAW must agree with what the message says, either way
        assertEquals(
                reply.text(),
                shown.toString(),
                "streamed fragments and the finished message must agree");
        return reply;
    }

    private static List<Part.ToolCall> calls(Message message) {
        return message.content().stream()
                .filter(Part.ToolCall.class::isInstance)
                .map(Part.ToolCall.class::cast)
                .toList();
    }

    @Test
    void offeredTools_callSyntaxBecomesAToolCall() {
        Message reply = feed(true, HELLO, CALL_OPEN, PAYLOAD, CALL_CLOSE);
        assertEquals(1, calls(reply).size(), "the span must be claimed when tools were offered");
        assertEquals("Hello", reply.text(), "a claimed call contributes no visible text");
    }

    @Test
    void noToolsOffered_theSameTokensStayVisibleText() {
        Message reply = feed(false, HELLO, CALL_OPEN, PAYLOAD, CALL_CLOSE);
        assertTrue(
                calls(reply).isEmpty(),
                "a caller who offered no tools must never receive a tool call");
        assertTrue(
                reply.text().contains("[f(x=1)]"),
                "the payload must remain readable text, not vanish: " + reply.text());
    }

    /** Reasoning is a lane, not a mode: it routes the same whether or not calls are claimed. */
    @Test
    void reasoningRoutesIndependentlyOfCallClaiming() {
        for (boolean claim : new boolean[] {true, false}) {
            ReplyLanes lanes =
                    new ReplyLanes(Optional.of(CLAIMING), ReplyParserTest.TOK, new int[0], claim);
            lanes.feed(THINK_OPEN);
            String thought = lanes.feed(HELLO);
            assertTrue(
                    lanes.reasoning(),
                    "inside the span, fragments are reasoning (claim=" + claim + ")");
            assertEquals("Hello", thought);
            lanes.feed(THINK_CLOSE);
            String answer = lanes.feed(HELLO);
            assertFalse(lanes.reasoning(), "after the close, fragments are content");
            assertEquals("Hello", answer);
        }
    }
}
