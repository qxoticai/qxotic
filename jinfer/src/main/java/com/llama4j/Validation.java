// Request validation for the OpenAI endpoints: shape, roles, tools, and sampling parameters.
// Runs on the HTTP handler thread (before queueing) so malformed requests fail fast with a 400.
package com.llama4j;

import com.llama4j.LFM25.Options;

import java.util.List;
import java.util.Map;

final class Validation {
    private Validation() {
    }

    static void validateChatRequest(Map<String, Object> request) {
        List<Object> messages = Values.asArray(request.get("messages"), "messages");
        Options.require(!messages.isEmpty(), "messages must not be empty");
        boolean substance = false;
        for (Object message : messages) {
            Map<String, Object> m = Values.asObject(message, "message");
            String role = Values.stringValue(m.get("role"), "");
            Options.require(List.of("system", "user", "assistant", "tool").contains(role),
                    "Invalid role: %s (must be system, user, assistant, or tool)", role);
            substance |= !Values.messageContent(m.get("content")).isBlank()
                    || (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty());
        }
        Options.require(substance, "messages must contain at least one non-empty message");
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?, ?> m) {
            String type = Values.stringValue(m.get("type"), "");
            Options.require("json_object".equals(type) || "text".equals(type),
                    "Unsupported response_format type: %s (only json_object and text are supported)", type);
            if ("json_object".equals(type)) {
                boolean hasJsonHint = false;
                for (Object message : messages) {
                    Map<String, Object> msg = Values.asObject(message, "message");
                    String role = Values.stringValue(msg.get("role"), "");
                    String content = Values.messageContent(msg.get("content"));
                    if (("system".equals(role) || "user".equals(role)) && content.toLowerCase().contains("json"))
                        hasJsonHint = true;
                }
                Options.require(hasJsonHint,
                        "response_format json_object requires the word 'json' in a system or user message");
            }
        }
        Object tools = request.get("tools");
        if (tools != null) {
            List<Object> toolList = Values.asArray(tools, "tools");
            for (Object value : toolList) validateTool(value);
        }
        Object toolChoice = request.get("tool_choice");
        if (toolChoice instanceof String s) {
            Options.require(List.of("auto", "none", "required").contains(s), "tool_choice must be auto, none, required, or a function choice object");
        } else if (toolChoice instanceof Map<?, ?> map) {
            Options.require("function".equals(map.get("type")), "Only function tool_choice objects are supported");
            Object function = map.get("function");
            Options.require(function instanceof Map<?, ?> fn && fn.get("name") instanceof String, "tool_choice.function.name is required");
        } else if (toolChoice != null) {
            throw new IllegalArgumentException("tool_choice must be a string or object");
        }
    }

    static void validateTool(Object value) {
        Map<String, Object> tool = Values.asObject(value, "tool");
        Options.require("function".equals(Values.stringValue(tool.get("type"), "function")), "Only function tools are supported");
        Map<String, Object> function = Values.asObject(tool.get("function"), "tool.function");
        Options.require(function.get("name") instanceof String name && !name.isBlank(), "tool.function.name is required");
    }

    /** Sampling-parameter validation shared by all endpoints; called on the HTTP handler thread
     *  (before queueing, and before any SSE headers) so invalid requests fail fast with a 400. */
    static void validateGenerationParams(Map<String, Object> request, Options options) {
        Options.require(request.get("model") instanceof String name && !name.isBlank(), "model is required");
        if (request.get("model") instanceof String name) {
            String served = options.modelPath().getFileName().toString();
            Options.require(name.equalsIgnoreCase(served), "Unknown model: %s (this server serves %s)", name, served);
        }
        if ((request.containsKey("grammar") || request.containsKey("response_format"))
                && options.noGrammar()) {
            Options.require(false, "Grammar constraints disabled (--no-grammar)");
        }
        Options.require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        float temperature = Values.floatValue(request.get("temperature"), options.temperature());
        Options.require(Float.isFinite(temperature) && 0 <= temperature && temperature <= 2, "Invalid argument: temperature must be within [0, 2]");
        float topp = Values.floatValue(request.get("top_p"), options.topp());
        Options.require(Float.isFinite(topp) && 0 <= topp && topp <= 1, "Invalid argument: top_p must be within [0, 1]");
        Options.require(0 <= Values.intValue(request.getOrDefault("max_tokens", request.get("max_completion_tokens")), options.maxTokens()), "Invalid argument: max_tokens must be non-negative");
        Options.require(-1 <= Values.intValue(request.get("reasoning_max_tokens"), -1), "Invalid argument: reasoning_max_tokens must be -1 (uncapped) or non-negative");
        Values.longValue(request.get("seed"), 0); // type check only
        Options.require(!request.containsKey("logprobs") && !request.containsKey("top_logprobs"),
                "logprobs is not supported");
        Options.require(!request.containsKey("logit_bias"), "logit_bias is not supported");
        Options.require(!request.containsKey("frequency_penalty") && !request.containsKey("presence_penalty"),
                "frequency_penalty and presence_penalty are not supported");
    }
}
