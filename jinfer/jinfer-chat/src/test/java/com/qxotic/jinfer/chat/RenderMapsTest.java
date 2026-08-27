package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;

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
        assertEquals("c1", response.get("tool_call_id"));
        assertEquals("calc", response.get("name"));
    }
}
