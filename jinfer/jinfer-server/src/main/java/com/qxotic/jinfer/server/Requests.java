package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * OpenAI request-shape adapters: maps the wire shapes of the various endpoints onto the single
 * internal form the generation pipeline consumes (a chat-message list, a prompt string, the served
 * model id). Pure mapping — no inference, no transport — so each endpoint handler stays a thin
 * {@code parse → adapt → generate → respond}.
 */
final class Requests {

    private Requests() {}

    /** The model id to echo back: the request's {@code model}, else the served file name. */
    static String modelId(Map<String, Object> request, LLMOptions options) {
        return Values.stringValue(
                request.get("model"), options.modelPath().getFileName().toString());
    }

    /** The /v1/completions prompt: a string, or a string array joined by newlines. */
    static String completionPrompt(Map<String, Object> request) {
        Object promptValue = request.get("prompt");
        return promptValue instanceof List<?> prompts
                ? prompts.stream().map(String::valueOf).collect(Collectors.joining("\n"))
                : Values.stringValue(promptValue, "");
    }

    // ---- /v1/responses -----------------------------------------------------

    /**
     * Folds Responses-API spellings onto the chat shape in place: {@code max_output_tokens} ->
     * {@code max_tokens}, and flat {@code {type:function,name,...}} tools -> nested {@code
     * function}.
     */
    static void normalizeResponse(Map<String, Object> request) {
        if (!request.containsKey("max_tokens") && request.containsKey("max_output_tokens")) {
            request.put("max_tokens", request.get("max_output_tokens"));
        }
        Object tools = request.get("tools");
        if (tools instanceof List<?> values) {
            List<Object> normalized = new ArrayList<>();
            for (Object value : values) normalized.add(normalizeResponseTool(value));
            request.put("tools", normalized);
        }
    }

    private static Object normalizeResponseTool(Object value) {
        Map<String, Object> tool = Values.asObject(value, "tool");
        if (tool.get("function") != null) return tool;
        if ("function".equals(tool.get("type")) && tool.get("name") != null) {
            Map<String, Object> function = new LinkedHashMap<>();
            function.put("name", tool.get("name"));
            if (tool.get("description") != null)
                function.put("description", tool.get("description"));
            function.put("parameters", tool.getOrDefault("parameters", Map.of()));
            return Map.of("type", "function", "function", function);
        }
        return tool;
    }

    /** The Responses-API {@code instructions} + {@code input} folded into a chat-message list. */
    static List<Object> responseInputMessages(Map<String, Object> request) {
        List<Object> messages = new ArrayList<>();
        String instructions = Values.stringValue(request.get("instructions"), null);
        if (instructions != null && !instructions.isBlank()) {
            messages.add(Map.of("role", "system", "content", instructions));
        }
        Object input = request.get("input");
        if (input instanceof String s) {
            messages.add(Map.of("role", "user", "content", s));
        } else if (input instanceof List<?> list) {
            for (Object item : list) addResponseInputMessage(messages, item);
        } else if (input != null) {
            addResponseInputMessage(messages, input);
        } else {
            throw new IllegalArgumentException("input is required");
        }
        return messages;
    }

    private static void addResponseInputMessage(List<Object> messages, Object item) {
        if (item instanceof String s) {
            messages.add(Map.of("role", "user", "content", s));
            return;
        }
        Map<String, Object> map = Values.asObject(item, "input item");
        String type = Values.stringValue(map.get("type"), "message");
        if ("function_call_output".equals(type)) {
            messages.add(
                    Map.of(
                            "role", "tool",
                            "name", Values.stringValue(map.get("call_id"), "tool"),
                            "content", Values.stringValue(map.get("output"), "")));
            return;
        }
        String role = Values.stringValue(map.get("role"), "user");
        messages.add(Map.of("role", role, "content", responseInputText(map.get("content"))));
    }

    private static String responseInputText(Object content) {
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object part : parts) {
                if (part instanceof String s) {
                    sb.append(s);
                } else if (part instanceof Map<?, ?> map) {
                    Object text = map.get("text");
                    if (text == null) text = map.get("input_text");
                    if (text == null) text = map.get("output_text");
                    if (text != null) sb.append(text);
                }
            }
            return sb.toString();
        }
        return Values.stringValue(content, "");
    }
}
