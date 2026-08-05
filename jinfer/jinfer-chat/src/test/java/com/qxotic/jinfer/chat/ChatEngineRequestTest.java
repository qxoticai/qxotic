package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

/**
 * {@link ChatEngine.Request} is a positional record with adjacent same-typed knobs, and its own
 * comment says why it validates: a transposed temperature and topP would otherwise sample
 * differently and silently. Every integration builds one of these positionally - the server,
 * langchain4j and Spring AI - so the guard is the only thing standing between a mis-ordered
 * argument and a request that quietly samples wrong.
 */
final class ChatEngineRequestTest {

    private static final List<Message> ONE_TURN = List.of(Message.user("hi"));
    private static final List<Tool> ONE_TOOL = List.of(new Tool("f", "{\"name\": \"f\"}"));

    private static ChatEngine.Request request(float temperature, float topP) {
        return new ChatEngine.Request(
                ONE_TURN,
                List.of(),
                true,
                -1,
                null,
                0L,
                temperature,
                topP,
                40,
                0.05f,
                42L,
                null,
                null,
                false,
                List.of(),
                null);
    }

    @Test
    void aTransposedTemperatureAndTopPIsRejected() {
        // topP is a probability mass; temperature is unbounded. Swapping a typical pair (0.7, 0.95)
        // is silently plausible, which is exactly why the range check exists
        request(0.7f, 0.95f); // the right way round
        assertThrows(IllegalArgumentException.class, () -> request(0.95f, 1.7f));
        assertThrows(IllegalArgumentException.class, () -> request(0.7f, 0f));
        assertThrows(IllegalArgumentException.class, () -> request(-0.1f, 0.95f));
    }

    @Test
    void anEmptyConversationIsRejected() {
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                List.of(), List.of(), true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f,
                                42L, null, null, false, List.of(), null));
    }

    @Test
    void negativeBudgetsAreRejectedButUnlimitedIsNot() {
        // -1 = the model's own maximum, for both the completion and the reasoning budget
        new ChatEngine.Request(
                ONE_TURN, List.of(), true, -1, -1, 0L, 0.7f, 0.95f, 40, 0.05f, 42L, null, null,
                false, List.of(), null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -2, null, 0L, 0.7f, 0.95f, 40, 0.05f,
                                42L, null, null, false, List.of(), null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, -2, 0L, 0.7f, 0.95f, 40, 0.05f, 42L,
                                null, null, false, List.of(), null));
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, null, -1L, 0.7f, 0.95f, 40, 0.05f,
                                42L, null, null, false, List.of(), null));
    }

    /**
     * The engine owns tool_choice arbitration now: a forced call needs offered tools, and a NAMED
     * choice must name one of them - the model would otherwise be "forced" into a call the caller
     * cannot dispatch.
     */
    @Test
    void aForcedToolMustBeOffered() {
        new ChatEngine.Request(
                ONE_TURN, ONE_TOOL, true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f, 42L, null, "",
                false, List.of(), null); // "" = any offered tool
        new ChatEngine.Request(
                ONE_TURN, ONE_TOOL, true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f, 42L, null, "f",
                false, List.of(), null);
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, List.of(), true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f,
                                42L, null, "", false, List.of(), null),
                "forcing with no tools offered");
        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChatEngine.Request(
                                ONE_TURN, ONE_TOOL, true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f, 42L,
                                null, "g", false, List.of(), null),
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
                                0.7f,
                                0.95f,
                                40,
                                0.05f,
                                42L,
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
        Map<String, Object> kwargs = new java.util.HashMap<>(Map.of("enable_thinking", false));
        ChatEngine.Request request =
                new ChatEngine.Request(
                        messages, List.of(), true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f, 42L, null,
                        null, false, stops, kwargs);

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
                        ONE_TURN, null, true, -1, null, 0L, 0.7f, 0.95f, 40, 0.05f, 42L, null, null,
                        false, null, null);
        assertTrue(request.tools().isEmpty());
        assertTrue(request.stops().isEmpty());
        // kwargs stays null: absent is not the same as an empty override set, and the Jinja render
        // distinguishes them
        assertNull(request.templateKwargs());
    }
}
