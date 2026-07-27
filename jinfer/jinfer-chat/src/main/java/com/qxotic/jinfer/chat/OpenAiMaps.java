package com.qxotic.jinfer.chat;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * The OpenAI wire shapes the Jinja whole-render fallback consumes, built in ONE place so every
 * framework integration feeds templates the identical map geometry - the templates key on these
 * exact field names, so a shape drift between integrations is a silent rendering bug.
 */
public final class OpenAiMaps {

    private OpenAiMaps() {}

    /**
     * {@code {type: function, function: {name, description?, parameters?}}} - one declared tool.
     */
    public static Map<String, Object> tool(String name, String description, Object parameters) {
        var fn = new LinkedHashMap<String, Object>();
        fn.put("name", name);
        if (description != null) fn.put("description", description);
        if (parameters != null) fn.put("parameters", parameters);
        var tool = new LinkedHashMap<String, Object>();
        tool.put("type", "function");
        tool.put("function", fn);
        return tool;
    }

    /**
     * {@code {id, type: function, function: {name, arguments}}} - one assistant {@code tool_calls}
     * entry (null id renders as "").
     */
    public static Map<String, Object> toolCall(String id, String name, String argumentsJson) {
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
    public static Map<String, Object> toolResponse(String content, String callId, String name) {
        var tool = new LinkedHashMap<String, Object>();
        tool.put("role", "tool");
        tool.put("content", content);
        tool.put("tool_call_id", callId);
        tool.put("name", name);
        return tool;
    }

    /** Parsed JSON tool arguments as a map; blank or non-object JSON = empty map. */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> args(String argumentsJson) {
        if (argumentsJson == null || argumentsJson.isBlank()) return Map.of();
        Object parsed = JsonCodec.parse(argumentsJson);
        return parsed instanceof Map ? (Map<String, Object>) parsed : Map.of();
    }
}
