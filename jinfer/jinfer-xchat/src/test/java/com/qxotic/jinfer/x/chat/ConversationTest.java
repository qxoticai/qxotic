package com.qxotic.jinfer.x.chat;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * The pairing law: every tool RESULT answers a tool CALL earlier in the same conversation. An
 * orphan renders as a ghost exchange the model never made - and the prompt cache commits it - so
 * construction refuses loudly instead. Unanswered calls are legal (the live hand-off to the
 * caller's executor).
 */
class ConversationTest {

    private static Message call(String id, String name) {
        return new Message(
                Role.ASSISTANT, List.of(new Content.ToolCall(id, name, Map.of("q", "2+2"), null)));
    }

    private static Message result(String id, String text) {
        return new Message(Role.TOOL, List.of(new Content.ToolResult(id, text)));
    }

    @Test
    void aMatchedRoundTripBuilds() {
        assertDoesNotThrow(
                () ->
                        new Conversation(
                                List.of(
                                        new Message(Role.USER, "what is 2+2?"),
                                        call("c1", "calc"),
                                        result("c1", "4"),
                                        new Message(Role.USER, "thanks"))));
    }

    @Test
    void unansweredCallsAreTheLiveHandOff() {
        // the engine hands the parsed calls to the caller: a conversation ENDING on unanswered
        // calls is the normal "now run the tools" state, never an error
        assertDoesNotThrow(
                () ->
                        new Conversation(
                                List.of(new Message(Role.USER, "call calc"), call("c1", "calc"))));
    }

    @Test
    void severalCallsPairWithTheirResultsInAnyOrder() {
        assertDoesNotThrow(
                () ->
                        new Conversation(
                                List.of(
                                        new Message(
                                                Role.ASSISTANT,
                                                List.of(
                                                        new Content.ToolCall(
                                                                "c1", "calc", Map.of(), null),
                                                        new Content.ToolCall(
                                                                "c2", "clock", Map.of(), null))),
                                        result("c2", "noon"),
                                        result("c1", "4"))));
    }

    @Test
    void anOrphanResultIsRefusedWithTheRecipe() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new Conversation(
                                        List.of(
                                                new Message(Role.USER, "what is 2+2?"),
                                                result("ghost", "4"))));
        assertTrue(e.getMessage().contains("ghost"), e.getMessage());
        assertTrue(e.getMessage().contains("no tool call"), e.getMessage());
    }

    @Test
    void aResultFromAnEvictedCallIsAnOrphanToo() {
        // the memory-trim artifact: the call fell off the window, the result stayed
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new Conversation(
                                        List.of(
                                                new Message(Role.USER, "hi"),
                                                call("c1", "calc"),
                                                result("c1", "4"),
                                                // ... the trim evicted the call above ...
                                                new Message(Role.USER, "again"),
                                                result("c1", "4"))));
        assertTrue(e.getMessage().contains("c1"), e.getMessage());
    }

    @Test
    void idLessResultsFollowAnyEarlierCall() {
        // families without call ids: presence of a call is the whole check
        assertDoesNotThrow(() -> new Conversation(List.of(call(null, "calc"), result(null, "4"))));
        assertThrows(
                IllegalArgumentException.class, () -> new Conversation(List.of(result(null, "4"))));
    }

    @Test
    void idLessCallsPairWithAdapterMintedResultIds() {
        // the langchain4j adapter mints positional "call_N" ids on id-less (pythonic) family
        // calls; the echoed result carries that minted id - presence of the id-less call is the
        // match, or every LFM2-class tool loop would refuse
        assertDoesNotThrow(
                () -> new Conversation(List.of(call("", "calc"), result("call_0", "4"))));
        // ...but the ghost lane stays shut: a minted id with NO call anywhere still refuses
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new Conversation(
                                        List.of(
                                                new Message(Role.USER, "hi"),
                                                result("call_0", "4"))));
        assertTrue(e.getMessage().contains("call_0"), e.getMessage());
    }

    @Test
    void appendKeepsTheLaw() {
        Conversation live =
                new Conversation(List.of(new Message(Role.USER, "call calc"), call("c1", "calc")));
        assertEquals(3, live.append(result("c1", "4")).messages().size());
        assertThrows(IllegalArgumentException.class, () -> live.append(result("nope", "4")));
    }
}
