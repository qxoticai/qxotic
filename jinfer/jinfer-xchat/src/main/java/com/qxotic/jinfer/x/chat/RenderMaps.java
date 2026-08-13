package com.qxotic.jinfer.x.chat;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Builds the OpenAI wire shapes the Jinja whole-render consumes, from the engine's own {@link
 * Conversation} - ONE place, so every template keys on identical map geometry (the templates read
 * these exact field names; a shape drift is a silent rendering bug). Media never reaches here: the
 * engine fails it loudly before the Jinja path.
 */
final class RenderMaps {

    private RenderMaps() {}

    static List<Object> messages(Conversation conversation) {
        List<Object> out = new ArrayList<>();
        for (Message message : conversation.messages()) {
            String role = message.role().name();
            if (Role.TOOL.equals(message.role())) {
                for (Content part : message.content()) {
                    if (part instanceof Content.ToolResult result) {
                        out.add(toolResponse(result.text(), result.callId(), null));
                    }
                }
                continue;
            }
            var map = new LinkedHashMap<String, Object>();
            map.put("role", role);
            map.put("content", text(message.content()));
            List<Map<String, Object>> calls = toolCalls(message.content());
            if (!calls.isEmpty()) map.put("tool_calls", calls);
            out.add(map);
        }
        return out;
    }

    static List<Object> tools(List<Tool> tools) {
        if (tools == null || tools.isEmpty()) return null;
        List<Object> out = new ArrayList<>(tools.size());
        for (Tool tool : tools) out.add(tool(tool.definition()));
        return out;
    }

    /**
     * The message's visible text: plain parts joined, reasoning wrapped in the think markers -
     * legal in echoed history (the scrub exempts them), which is how templates that split on {@code
     * </think>} see their own past reasoning.
     */
    private static String text(List<Content> parts) {
        StringBuilder out = new StringBuilder();
        for (Content part : parts) {
            if (part instanceof Content.Text text) {
                out.append(text.text());
            } else if (part instanceof Content.Reasoning reasoning) {
                out.append("<think>");
                out.append(text(reasoning.content()));
                out.append("</think>");
            } else if (part instanceof Content.Media) {
                throw new UnsupportedOperationException(
                        "media content cannot take the Jinja whole-render path");
            }
        }
        return out.toString();
    }

    private static List<Map<String, Object>> toolCalls(List<Content> parts) {
        List<Map<String, Object>> out = new ArrayList<>();
        for (Content part : parts) {
            if (part instanceof Content.ToolCall call) {
                out.add(toolCall(call.id(), call.name(), JsonCodec.stringify(call.arguments())));
            }
        }
        return out;
    }

    /** {@code {type: function, function: {...}}} - one declared tool. */
    private static Map<String, Object> tool(Map<String, Object> definition) {
        if (definition.containsKey("function")) return definition;
        var tool = new LinkedHashMap<String, Object>();
        tool.put("type", "function");
        tool.put("function", definition);
        return tool;
    }

    /**
     * {@code {id, type: function, function: {name, arguments}}} - one assistant tool_calls entry.
     */
    private static Map<String, Object> toolCall(String id, String name, String argumentsJson) {
        var call = new LinkedHashMap<String, Object>();
        call.put("id", id == null ? "" : id);
        call.put("type", "function");
        var fn = new LinkedHashMap<String, Object>();
        fn.put("name", name);
        fn.put("arguments", argumentsJson);
        call.put("function", fn);
        return call;
    }

    /** {@code {role: tool, content, tool_call_id, name}} - one tool-response message. */
    private static Map<String, Object> toolResponse(String content, String callId, String name) {
        var tool = new LinkedHashMap<String, Object>();
        tool.put("role", "tool");
        tool.put("content", content);
        tool.put("tool_call_id", callId);
        tool.put("name", name != null ? name : callId);
        return tool;
    }
}
