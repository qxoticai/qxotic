package com.qxotic.jinfer.chat;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class JinjaChatTemplateTest {

    private static final List<String> NAMES = List.of("<|im_start|>", "<|im_end|>");

    @Test
    void scrubBreaksEmbeddedSpecialTokenStrings() {
        String scrubbed = JinjaChatTemplate.scrub("say <|im_start|> twice <|im_start|>", NAMES);
        assertFalse(scrubbed.contains("<|im_start|>"), "no intact special spelling survives");
        assertEquals("say <\u200b|im_start|> twice <\u200b|im_start|>", scrubbed);
    }

    @Test
    void scrubValueKeepsCleanGraphsIdentical() {
        Map<String, Object> clean = Map.of("role", "user", "content", "hello");
        assertSame(
                clean, JinjaChatTemplate.scrubValue(clean, NAMES), "clean graphs allocate nothing");
    }

    @Test
    void scrubValueScrubsDeepStringsButNotKeys() {
        Map<String, Object> dirty =
                Map.of("<|im_start|>key", List.of("text with <|im_end|> inside"));
        @SuppressWarnings("unchecked")
        Map<String, Object> scrubbed =
                (Map<String, Object>) JinjaChatTemplate.scrubValue(dirty, NAMES);
        assertTrue(scrubbed.containsKey("<|im_start|>key"), "keys pass through untouched");
        assertFalse(scrubbed.get("<|im_start|>key").toString().contains("<|im_end|>"));
    }

    @Test
    void mapsGeometryIsTheOpenAiWire() {
        Conversation conversation =
                new Conversation(
                        List.of(
                                new Message(Role.SYSTEM, "be terse"),
                                new Message(Role.USER, "call me maybe"),
                                new Message(
                                        Role.ASSISTANT,
                                        List.of(
                                                new Content.Text("calling", null),
                                                new Content.ToolCall(
                                                        "c1", "dial", Map.of("number", 42), null))),
                                new Message(
                                        Role.TOOL, List.of(new Content.ToolResult("c1", "busy")))),
                        List.of(new Tool("dial", Map.of("name", "dial"))),
                        false,
                        "");
        List<Object> messages = RenderMaps.messages(conversation);
        assertEquals(4, messages.size());
        @SuppressWarnings("unchecked")
        Map<String, Object> assistant = (Map<String, Object>) messages.get(2);
        assertEquals("calling", assistant.get("content"));
        @SuppressWarnings("unchecked")
        List<Map<String, Object>> calls = (List<Map<String, Object>>) assistant.get("tool_calls");
        assertEquals("c1", calls.get(0).get("id"));
        @SuppressWarnings("unchecked")
        Map<String, Object> fn = (Map<String, Object>) calls.get(0).get("function");
        assertEquals("dial", fn.get("name"));
        assertEquals("{\"number\":42}", fn.get("arguments"));
        @SuppressWarnings("unchecked")
        Map<String, Object> toolResult = (Map<String, Object>) messages.get(3);
        assertEquals("tool", toolResult.get("role"));
        assertEquals("busy", toolResult.get("content"));
        assertEquals("c1", toolResult.get("tool_call_id"));

        List<Object> tools = RenderMaps.tools(conversation.tools());
        @SuppressWarnings("unchecked")
        Map<String, Object> tool = (Map<String, Object>) tools.get(0);
        assertEquals("function", tool.get("type"));
    }
}
