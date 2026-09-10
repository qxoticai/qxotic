package com.qxotic.jinfer.server;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import com.qxotic.jinfer.chat.Content;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class ToolCallsTest {

    @Test
    @SuppressWarnings("unchecked")
    void structuredCallsMapWithoutReparsingModelText() {
        var wire =
                ToolCalls.toWire(
                                List.of(
                                        new Content.ToolCall(
                                                "call-1", "weather", Map.of("city", "Paris"))))
                        .getFirst();
        assertEquals("call-1", wire.get("id"));
        Map<String, Object> function = (Map<String, Object>) wire.get("function");
        assertEquals("weather", function.get("name"));
        assertTrue(((String) function.get("arguments")).contains("Paris"));
    }
}
