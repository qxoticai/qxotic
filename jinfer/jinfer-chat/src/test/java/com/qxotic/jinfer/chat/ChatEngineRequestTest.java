package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.cache.PromptCache;
import com.qxotic.jinfer.llm.Generator;
import com.qxotic.jinfer.llm.Sampling;
import java.time.Duration;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * {@link ChatEngine.Request} is a positional record built by every integration, so its guards are
 * the only thing standing between a mis-ordered argument and a request that quietly runs wrong.
 */
final class ChatEngineRequestTest {

    private static final List<Message> ONE_TURN = List.of(Message.user("hi"));
    private static final List<Tool> ONE_TOOL = List.of(new Tool("f", Map.of("name", "f")));
    private static final Sampling SAMPLING = new Sampling(0.7f, 0.95f, 40, 0.05f, 42L);

    @Test
    void ofGivesTheConservativeDefaults() {
        ChatEngine.Request r = ChatEngine.Request.of(ONE_TURN, SAMPLING);
        assertEquals(ONE_TURN, r.messages());
        assertEquals(SAMPLING, r.sampling());
        assertTrue(r.tools().isEmpty());
        assertFalse(r.thinking());
        assertEquals(Generator.Constraints.UNLIMITED, r.maxTokens());
        assertNull(r.reasoningMaxTokens());
        assertNull(r.reasoningMessage());
        assertEquals(Duration.ZERO, r.timeout());
        assertNull(r.contentGbnf());
        assertEquals(ChatEngine.ForcedTool.NONE, r.forcedTool());
        assertTrue(r.stops().isEmpty());
        assertNull(r.templateKwargs());
    }

    @Test
    void anEmptyConversationIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                List.of(),
                                List.of(),
                                true,
                                -1,
                                null,
                                null,
                                Duration.ZERO,
                                SAMPLING,
                                null,
                                null,
                                List.of(),
                                null));
    }

    @Test
    void samplingIsRequired() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -1,
                                null,
                                null,
                                Duration.ZERO,
                                null,
                                null,
                                null,
                                List.of(),
                                null));
    }

    @Test
    void negativeBudgetsAreRejectedButUnlimitedIsNot() {
        // -1 = the model's own maximum, for both the completion and the reasoning budget
        new ChatEngine.Request(
                ONE_TURN,
                List.of(),
                true,
                -1,
                -1,
                null,
                Duration.ZERO,
                SAMPLING,
                null,
                null,
                List.of(),
                null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -2,
                                null,
                                null,
                                Duration.ZERO,
                                SAMPLING,
                                null,
                                null,
                                List.of(),
                                null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -1,
                                -2,
                                null,
                                Duration.ZERO,
                                SAMPLING,
                                null,
                                null,
                                List.of(),
                                null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -1,
                                null,
                                null,
                                Duration.ofNanos(-1),
                                SAMPLING,
                                null,
                                null,
                                List.of(),
                                null));
    }

    /**
     * The engine owns tool_choice arbitration: a forced call needs offered tools, and a NAMED
     * choice must name one of them - the model would otherwise be "forced" into a call the caller
     * cannot dispatch.
     */
    @Test
    void aForcedToolMustBeOffered() {
        new ChatEngine.Request(
                ONE_TURN,
                ONE_TOOL,
                true,
                -1,
                null,
                null,
                Duration.ZERO,
                SAMPLING,
                null,
                ChatEngine.ForcedTool.ANY,
                List.of(),
                null);
        new ChatEngine.Request(
                ONE_TURN,
                ONE_TOOL,
                true,
                -1,
                null,
                null,
                Duration.ZERO,
                SAMPLING,
                null,
                new ChatEngine.ForcedTool.Named("f"),
                List.of(),
                null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -1,
                                null,
                                null,
                                Duration.ZERO,
                                SAMPLING,
                                null,
                                ChatEngine.ForcedTool.ANY,
                                List.of(),
                                null),
                "forcing with no tools offered");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                ONE_TOOL,
                                true,
                                -1,
                                null,
                                null,
                                Duration.ZERO,
                                SAMPLING,
                                null,
                                new ChatEngine.ForcedTool.Named("g"),
                                List.of(),
                                null),
                "forcing a tool that was never offered");
    }

    @Test
    void anAbsentForcedToolMeansNone() {
        ChatEngine.Request request =
                new ChatEngine.Request(
                        ONE_TURN,
                        List.of(),
                        true,
                        -1,
                        null,
                        null,
                        Duration.ZERO,
                        SAMPLING,
                        null,
                        null,
                        List.of(),
                        null);
        assertEquals(ChatEngine.ForcedTool.NONE, request.forcedTool());
    }

    /** Callers hand over live collections; a request that mutates under the engine is a bug. */
    @Test
    void theRequestDefensivelyCopiesWhatItIsGiven() {
        List<Message> messages = new ArrayList<>(ONE_TURN);
        List<String> stops = new ArrayList<>(List.of("STOP"));
        Map<String, Object> kwargs = new HashMap<>(Map.of("enable_thinking", false));
        ChatEngine.Request request =
                new ChatEngine.Request(
                        messages,
                        List.of(),
                        true,
                        -1,
                        null,
                        null,
                        Duration.ZERO,
                        SAMPLING,
                        null,
                        null,
                        stops,
                        kwargs);

        messages.add(Message.user("smuggled"));
        stops.add("SMUGGLED");
        kwargs.put("smuggled", true);

        assertEquals(1, request.messages().size(), "messages must be copied");
        assertEquals(List.of("STOP"), request.stops(), "stops must be copied");
        assertEquals(1, request.templateKwargs().size(), "template kwargs must be copied");
    }

    @Test
    void absentCollectionsBecomeEmptyRatherThanNull() {
        ChatEngine.Request request =
                new ChatEngine.Request(
                        ONE_TURN,
                        null,
                        true,
                        -1,
                        null,
                        null,
                        Duration.ZERO,
                        SAMPLING,
                        null,
                        null,
                        null,
                        null);
        assertTrue(request.tools().isEmpty());
        assertTrue(request.stops().isEmpty());
        // kwargs stays null: absent is not the same as an empty override set, and the Jinja
        // render distinguishes them
        assertNull(request.templateKwargs());
    }

    /** A delta is one fragment: no channel-less or empty emission can be constructed. */
    @Test
    void aDeltaIsNeverEmpty() {
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatEngine.Delta(null, "x", com.qxotic.toknroll.IntSequence.of(1)));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Delta(
                                Channel.CONTENT, "", com.qxotic.toknroll.IntSequence.of(1)));
        assertThrows(
                IllegalArgumentException.class,
                () -> new ChatEngine.Delta(Channel.CONTENT, "x", null));
    }

    /** Cancelled is derived from the absent reply - there is no boolean to disagree with. */
    @Test
    void aCancelledCompletionIsExactlyOneWithoutAReply() {
        assertTrue(
                new ChatEngine.Completion(null, null, false, 0, 0, PromptCache.Tier.FRESH, null)
                        .cancelled());
        assertFalse(
                new ChatEngine.Completion(
                                Message.assistant("hi"),
                                null,
                                false,
                                0,
                                0,
                                PromptCache.Tier.FRESH,
                                null)
                        .cancelled());
    }

    @Test
    void aForcedCallAndAContentGrammarCannotBothBeTheReply() {
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () ->
                                new ChatEngine.Request(
                                        ONE_TURN,
                                        ONE_TOOL,
                                        true,
                                        -1,
                                        null,
                                        null,
                                        Duration.ZERO,
                                        SAMPLING,
                                        "root ::= \"x\"",
                                        ChatEngine.ForcedTool.ANY,
                                        List.of(),
                                        null));
        assertTrue(e.getMessage().contains("both"), e.getMessage());
    }
}
