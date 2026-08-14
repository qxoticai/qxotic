package com.qxotic.jinfer.x.server;

import com.qxotic.jinfer.x.chat.Content;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

/** Structured engine tool calls mapped to the OpenAI wire. */
final class ToolCalls {
    private ToolCalls() {}

    static List<Map<String, Object>> toWire(List<Content.ToolCall> calls) {
        List<Map<String, Object>> out = new ArrayList<>(calls.size());
        for (Content.ToolCall call : calls) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", call.name());
            function.put("arguments", JsonCodec.stringify(call.arguments()));
            Map<String, Object> wire = new LinkedHashMap<>();
            wire.put("id", call.id().isEmpty() ? "call_" + UUID.randomUUID() : call.id());
            wire.put("type", "function");
            wire.put("function", function);
            out.add(wire);
        }
        return out;
    }

    static List<Map<String, Object>> toolCallDeltas(List<Map<String, Object>> calls) {
        List<Map<String, Object>> out = new ArrayList<>(calls.size());
        for (int i = 0; i < calls.size(); i++) {
            Map<String, Object> delta = new LinkedHashMap<>(calls.get(i));
            delta.put("index", i);
            out.add(delta);
        }
        return out;
    }
}
