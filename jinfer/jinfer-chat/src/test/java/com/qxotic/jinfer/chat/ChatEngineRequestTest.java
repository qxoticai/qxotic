package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.llm.Sampling;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * {@link ChatEngine.Request} is a positional record built by three integrations - the server,
 * langchain4j and Spring AI - so its guards are the only thing standing between a mis-ordered
 * argument and a request that quietly runs wrong. The four sampling knobs used to sit here loose
 * and adjacent and are now one {@link Sampling}, which validates its own ranges (see {@code
 * SamplingTest}); what remains here is everything a range check cannot express.
 */
final class ChatEngineRequestTest {

    private static final List<Message> ONE_TURN = List.of(Message.user("hi"));
    private static final List<Tool> ONE_TOOL = List.of(new Tool("f", "{\"name\": \"f\"}"));
    private static final Sampling SAMPLING = new Sampling(0.7f, 0.95f, 40, 0.05f, 42L);

    @Test
    void anEmptyConversationIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                List.of(), List.of(), true, -1, null, 0L, SAMPLING, null, null,
                                false, List.of(), null));
    }

    @Test
    void samplingIsRequired() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, null, 0L, null, null, null, false,
                                List.of(), null));
    }

    @Test
    void negativeBudgetsAreRejectedButUnlimitedIsNot() {
        // -1 = the model's own maximum, for both the completion and the reasoning budget
        new ChatEngine.Request(
                ONE_TURN, List.of(), true, -1, -1, 0L, SAMPLING, null, null, false, List.of(),
                null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -2, null, 0L, SAMPLING, null, null,
                                false, List.of(), null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, -2, 0L, SAMPLING, null, null, false,
                                List.of(), null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, null, -1L, SAMPLING, null, null,
                                false, List.of(), null));
    }

    /**
     * The engine owns tool_choice arbitration now: a forced call needs offered tools, and a NAMED
     * choice must name one of them - the model would otherwise be "forced" into a call the caller
     * cannot dispatch.
     */
    @Test
    void aForcedToolMustBeOffered() {
        new ChatEngine.Request(
                ONE_TURN, ONE_TOOL, true, -1, null, 0L, SAMPLING, null, "", false, List.of(),
                null); // "" = any offered tool
        new ChatEngine.Request(
                ONE_TURN, ONE_TOOL, true, -1, null, 0L, SAMPLING, null, "f", false, List.of(),
                null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, null, 0L, SAMPLING, null, "", false,
                                List.of(), null),
                "forcing with no tools offered");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, ONE_TOOL, true, -1, null, 0L, SAMPLING, null, "g", false,
                                List.of(), null),
                "forcing a tool that was never offered");
    }

    /** A view is native-only and the native codec never sees kwargs - contradictory by law. */
    @Test
    void aCachedViewCannotCarryTemplateKwargs() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN,
                                List.of(),
                                true,
                                -1,
                                null,
                                0L,
                                SAMPLING,
                                null,
                                null,
                                true,
                                List.of(),
                                Map.of("custom_var", 1)));
    }

    /** Callers hand over live collections; a request that mutates under the engine is a bug. */
    @Test
    void theRequestDefensivelyCopiesWhatItIsGiven() {
        List<Message> messages = new ArrayList<>(ONE_TURN);
        List<String> stops = new ArrayList<>(List.of("STOP"));
        Map<String, Object> kwargs = new HashMap<>(Map.of("enable_thinking", false));
        ChatEngine.Request request =
                new ChatEngine.Request(
                        messages, List.of(), true, -1, null, 0L, SAMPLING, null, null, false, stops,
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
                        ONE_TURN, null, true, -1, null, 0L, SAMPLING, null, null, false, null,
                        null);
        assertTrue(request.tools().isEmpty());
        assertTrue(request.stops().isEmpty());
        // kwargs stays null: absent is not the same as an empty override set, and the Jinja render
        // distinguishes them
        assertNull(request.templateKwargs());
    }
}
