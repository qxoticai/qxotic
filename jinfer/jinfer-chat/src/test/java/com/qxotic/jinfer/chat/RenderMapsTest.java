package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RenderMapsTest {

    @Test
    @SuppressWarnings("unchecked")
    void aToolResponseCarriesTheFunctionItAnswers() {
        // templates that key tool responses on the function name (minimax, granite) must see
        // the name of the call the result answers, matched through the call id
        Conversation conversation =
                new Conversation(
                        List.of(
                                new Message(Role.USER, "what is 2+2?"),
                                new Message(
                                        Role.ASSISTANT,
                                        List.of(
                                                new Content.ToolCall(
                                                        "c1", "calc", Map.of("q", "2+2"), null))),
                                new Message(
                                        Role.TOOL, List.of(new Content.ToolResult("c1", "4")))));
        Map<String, Object> response =
                (Map<String, Object>) RenderMaps.messages(conversation).get(2);
        assertEquals("tool", response.get("role"));
        assertEquals(RenderMaps.promptId("c1"), response.get("tool_call_id"));
        assertEquals("calc", response.get("name"));
    }

    @Test
    void promptIdsAreNineAlphanumericsAndStable() {
        // Mistral's templates raise on anything else; the wire id is untouched (Conversation)
        assertEquals("abcdefghi", RenderMaps.promptId("abcdefghi"));
        for (String id : List.of("", "call_0", "call_9f2b1c7e-1234-4b1a-9f3e-abcdef012345")) {
            assertTrue(RenderMaps.promptId(id).matches("[a-f0-9]{9}"), id);
            assertEquals(RenderMaps.promptId(id), RenderMaps.promptId(id));
        }
        assertNotEquals(RenderMaps.promptId("call_0"), RenderMaps.promptId("call_1"));
    }
}
