// Request validation for the OpenAI endpoints: shape, roles, tools, and sampling parameters.
// Runs on the HTTP handler thread (before queueing) so malformed requests fail fast with a 400.
package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.llm.*;
import com.qxotic.jinfer.llm.Sampling;
import java.util.List;
import java.util.Map;

final class Validation {
    private Validation() {}

    /**
     * A message whose content array carries any TYPED non-text part is substance too (an image-only
     * message has legitimately empty text) - and letting unknown types through here means the
     * request fails later with the precise "unsupported content part type" error instead of a
     * misleading "messages must not be empty".
     */
    private static boolean hasImageItem(Object content) {
        return content instanceof List<?> parts
                && parts.stream()
                        .anyMatch(
                                p ->
                                        p instanceof Map<?, ?> pm
                                                && pm.get("type") != null
                                                && !"text".equals(pm.get("type")));
    }

    static void validateChatRequest(Map<String, Object> request) {
        List<Object> messages = Values.asArray(request.get("messages"), "messages");
        require(!messages.isEmpty(), "messages must not be empty");
        boolean substance = false;
        for (Object message : messages) {
            Map<String, Object> m = Values.asObject(message, "message");
            String role = Values.stringValue(m.get("role"), "");
            require(
                    List.of("system", "user", "assistant", "tool").contains(role),
                    "Invalid role: %s (must be system, user, assistant, or tool)",
                    role);
            substance |=
                    !Values.messageContent(m.get("content")).isBlank()
                            || (m.get("tool_calls") instanceof List<?> calls && !calls.isEmpty())
                            || hasImageItem(m.get("content"));
        }
        require(substance, "messages must contain at least one non-empty message");
        Object fmt = request.get("response_format");
        if (fmt instanceof Map<?, ?> m) {
            String type = Values.stringValue(m.get("type"), "");
            require(
                    "json_object".equals(type) || "text".equals(type),
                    "Unsupported response_format type: %s (only json_object and text are"
                            + " supported)",
                    type);
            if ("json_object".equals(type)) {
                boolean hasJsonHint = false;
                for (Object message : messages) {
                    Map<String, Object> msg = Values.asObject(message, "message");
                    String role = Values.stringValue(msg.get("role"), "");
                    String content = Values.messageContent(msg.get("content"));
                    if (("system".equals(role) || "user".equals(role))
                            && content.toLowerCase().contains("json")) hasJsonHint = true;
                }
                require(
                        hasJsonHint,
                        "response_format json_object requires the word 'json' in a system or user"
                                + " message");
            }
        }
        Object tools = request.get("tools");
        if (tools != null) {
            List<Object> toolList = Values.asArray(tools, "tools");
            for (Object value : toolList) validateTool(value);
        }
        Object toolChoice = request.get("tool_choice");
        if (toolChoice instanceof String s) {
            require(
                    List.of("auto", "none", "required").contains(s),
                    "tool_choice must be auto, none, required, or a function choice object");
        } else if (toolChoice instanceof Map<?, ?> map) {
            require(
                    "function".equals(map.get("type")),
                    "Only function tool_choice objects are supported");
            Object function = map.get("function");
            require(
                    function instanceof Map<?, ?> fn && fn.get("name") instanceof String,
                    "tool_choice.function.name is required");
        } else if (toolChoice != null) {
            throw new IllegalArgumentException("tool_choice must be a string or object");
        }
    }

    static void validateTool(Object value) {
        Map<String, Object> tool = Values.asObject(value, "tool");
        require(
                "function".equals(Values.stringValue(tool.get("type"), "function")),
                "Only function tools are supported");
        Map<String, Object> function = Values.asObject(tool.get("function"), "tool.function");
        require(
                function.get("name") instanceof String name && !name.isBlank(),
                "tool.function.name is required");
    }

    /**
     * Rejects a malformed REQUEST. The message travels to the client in a 400 envelope, so it says
     * what is wrong with the request and never what the server is called or how it is configured.
     * The CLI has its own copy for argv, whose failure mode is different: usage text and exit 1.
     */
    static void require(boolean condition, String messageFormat, Object... args) {
        if (!condition) {
            throw new IllegalArgumentException(messageFormat.formatted(args));
        }
    }

    /**
     * Sampling-parameter validation shared by all endpoints; called on the HTTP handler thread
     * (before queueing, and before any SSE headers) so invalid requests fail fast with a 400.
     */
    static void validateGenerationParams(Map<String, Object> request, ServerConfig config) {
        Sampling sampling = config.defaults().sampling();
        require(
                request.get("model") instanceof String name && !name.isBlank(),
                "model is required");
        if (request.get("model") instanceof String name) {
            require(
                    name.equalsIgnoreCase(config.modelName()),
                    "Unknown model: %s (this server serves %s)",
                    name,
                    config.modelName());
        }
        if ((request.containsKey("grammar") || request.containsKey("response_format"))
                && !config.limits().grammar()) {
            require(false, "Grammar constraints disabled (--no-grammar)");
        }
        require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        float temperature = Values.floatValue(request.get("temperature"), sampling.temperature());
        require(
                Float.isFinite(temperature) && 0 <= temperature && temperature <= 2,
                "Invalid argument: temperature must be within [0, 2]");
        float topp = Values.floatValue(request.get("top_p"), sampling.topP());
        require(
                Float.isFinite(topp) && 0 <= topp && topp <= 1,
                "Invalid argument: top_p must be within [0, 1]");
        require(
                Values.intValue(request.get("top_k"), sampling.topK()) >= 0,
                "Invalid argument: top_k must be non-negative (0 disables it)");
        float minp = Values.floatValue(request.get("min_p"), sampling.minP());
        require(
                Float.isFinite(minp) && 0 <= minp && minp <= 1,
                "Invalid argument: min_p must be within [0, 1]");
        require(
                0
                        <= Values.intValue(
                                request.getOrDefault(
                                        "max_tokens", request.get("max_completion_tokens")),
                                config.defaults().maxOutputTokens()),
                "Invalid argument: max_tokens must be non-negative");
        require(
                -1 <= Values.intValue(request.get("reasoning_max_tokens"), -1),
                "Invalid argument: reasoning_max_tokens must be -1 (uncapped) or non-negative");
        Values.longValue(request.get("seed"), 0); // type check only
        require(
                !request.containsKey("logprobs") && !request.containsKey("top_logprobs"),
                "logprobs is not supported");
        require(!request.containsKey("logit_bias"), "logit_bias is not supported");
        require(
                !request.containsKey("frequency_penalty")
                        && !request.containsKey("presence_penalty"),
                "frequency_penalty and presence_penalty are not supported");
    }
}
