package com.qxotic.jinfer.x.server;

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

    /**
     * The model id to echo back: the request's {@code model}, else the served file name. Blank
     * counts as absent - echoing {@code "model": ""} back at a client is never the useful answer,
     * and {@link Validation#validateGenerationParams} lets blank through for that reason.
     */
    static String modelId(Map<String, Object> request, String servedModel) {
        String requested = Values.stringValue(request.get("model"), "");
        return requested.isBlank() ? servedModel : requested;
    }

    /**
     * The request's completion budget, whichever of the two spellings carries it, or null when
     * neither does. {@code getOrDefault("max_tokens", ...)} was wrong for both callers: it returns
     * the STORED null for a present-but-null key, so {@code {"max_tokens": null,
     * "max_completion_tokens": 100}} - what an OpenAI SDK serialises when only the newer field is
     * set - resolved to the server default and threw the client's 100 away.
     */
    static Object budget(Map<String, Object> request) {
        Object legacy = request.get("max_tokens");
        return legacy != null ? legacy : request.get("max_completion_tokens");
    }

    /** The /v1/completions prompt: a string, or a string array joined by newlines. */
    static String completionPrompt(Map<String, Object> request) {
        Object promptValue = request.get("prompt");
        if (promptValue instanceof String prompt) return prompt;
        if (promptValue instanceof List<?> prompts) {
            Validation.require(
                    prompts.stream().allMatch(String.class::isInstance),
                    "prompt array must contain only strings");
            return prompts.stream().map(String.class::cast).collect(Collectors.joining("\n"));
        }
        if (promptValue == null) return "";
        throw new IllegalArgumentException("prompt must be a string or an array of strings");
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
        if (!request.containsKey("response_format") && request.get("text") != null) {
            Map<String, Object> text = Values.asObject(request.get("text"), "text");
            if (text.get("format") != null) {
                request.put(
                        "response_format",
                        normalizeResponseFormat(
                                Values.asObject(text.get("format"), "text.format")));
            }
        }
        Object tools = request.get("tools");
        if (tools instanceof List<?> values) {
            List<Object> normalized = new ArrayList<>();
            for (Object value : values) normalized.add(normalizeResponseTool(value));
            request.put("tools", normalized);
        }
        Object choice = request.get("tool_choice");
        if (choice instanceof Map<?, ?> map
                && "function".equals(map.get("type"))
                && map.get("name") instanceof String name
                && map.get("function") == null) {
            request.put(
                    "tool_choice",
                    Map.of("type", "function", "function", Map.of("name", name)));
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

    private static Map<String, Object> normalizeResponseFormat(Map<String, Object> format) {
        if (!"json_schema".equals(format.get("type")) || format.get("json_schema") != null) {
            return format;
        }
        Map<String, Object> schema = new LinkedHashMap<>();
        for (String key : List.of("name", "description", "schema", "strict")) {
            if (format.get(key) != null) schema.put(key, format.get(key));
        }
        return Map.of("type", "json_schema", "json_schema", schema);
    }

    /** The Responses-API {@code instructions} + {@code input} folded into a chat-message list. */
    static List<Object> responseInputMessages(Map<String, Object> request) {
        List<Object> messages = new ArrayList<>();
        Object rawInstructions = request.get("instructions");
        Validation.require(
                rawInstructions == null || rawInstructions instanceof String,
                "instructions must be a string");
        String instructions = (String) rawInstructions;
        if (instructions != null && !instructions.isBlank()) {
            messages.add(Map.of("role", "system", "content", instructions));
        }
        Object input = request.get("input");
        if (input instanceof String s) {
            // blank is not input: an all-whitespace string used to become a user turn, which made
            // the endpoint's "input must not be empty" check pass and the model generate from
            // nothing - the very thing the chat endpoint's substance rule exists to prevent
            if (!s.isBlank()) messages.add(Map.of("role", "user", "content", s));
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
        if ("function_call".equals(type)) {
            Validation.require(
                    map.get("call_id") instanceof String callId && !callId.isBlank(),
                    "function_call.call_id is required");
            Validation.require(
                    map.get("name") instanceof String name && !name.isBlank(),
                    "function_call.name is required");
            Validation.require(
                    map.get("arguments") instanceof String,
                    "function_call.arguments must be a string");
            String callId = (String) map.get("call_id");
            String name = (String) map.get("name");
            messages.add(
                    Map.of(
                            "role",
                            "assistant",
                            "content",
                            "",
                            "tool_calls",
                            List.of(
                                    Map.of(
                                            "id",
                                            callId,
                                            "type",
                                            "function",
                                            "function",
                                            Map.of(
                                                    "name",
                                                    name,
                                                    "arguments",
                                                    map.get("arguments"))))));
            return;
        }
        if ("function_call_output".equals(type)) {
            Validation.require(
                    map.get("call_id") instanceof String callId && !callId.isBlank(),
                    "function_call_output.call_id is required");
            String callId = (String) map.get("call_id");
            messages.add(
                    Map.of(
                            "role", "tool",
                            "tool_call_id", callId,
                            "content", responseToolOutput(map.get("output"))));
            return;
        }
        String role = Values.stringValue(map.get("role"), "user");
        messages.add(Map.of("role", role, "content", responseInputContent(map.get("content"))));
    }

    private static Object responseToolOutput(Object output) {
        if (output instanceof String) return output;
        Validation.require(output instanceof List<?>, "function_call_output.output is required");
        Object normalized = responseInputContent(output);
        for (Object part : (List<?>) normalized) {
            Map<String, Object> value = Values.asObject(part, "function_call_output content");
            Validation.require(
                    "text".equals(value.get("type")),
                    "media function_call_output is not supported");
        }
        return normalized;
    }

    /** Normalize Responses text spellings while preserving media parts for the shared chat path. */
    private static Object responseInputContent(Object content) {
        if (content instanceof List<?> parts) {
            List<Object> normalized = new ArrayList<>(parts.size());
            for (Object part : parts) {
                if (part instanceof String s) {
                    normalized.add(Map.of("type", "text", "text", s));
                } else {
                    Map<String, Object> value = Values.asObject(part, "content part");
                    String type = Values.stringValue(value.get("type"), "");
                    if ("input_text".equals(type) || "output_text".equals(type)) {
                        normalized.add(
                                Map.of(
                                        "type",
                                        "text",
                                        "text",
                                        Values.stringValue(value.get("text"), "")));
                    } else {
                        normalized.add(value);
                    }
                }
            }
            return normalized;
        }
        return Values.stringValue(content, "");
    }
}
