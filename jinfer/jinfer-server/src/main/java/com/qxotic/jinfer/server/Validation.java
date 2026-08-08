// Request validation for the OpenAI endpoints: shape, roles, tools, and sampling parameters.
// Runs on the HTTP handler thread (before queueing) so malformed requests fail fast with a 400.
package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import com.qxotic.jinfer.chat.Values;
import com.qxotic.jinfer.llm.*;
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
        validateChatRequest(request, Values.asArray(request.get("messages"), "messages"));
    }

    /**
     * As above over turns the caller already folded - /v1/responses builds its message list from
     * {@code input} and must be held to the same rules, without a synthetic copy of the request
     * carrying a {@code messages} key it never had.
     */
    static void validateChatRequest(Map<String, Object> request, List<Object> messages) {
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
                    "json_object".equals(type) || "json_schema".equals(type) || "text".equals(type),
                    "Unsupported response_format type: %s (only json_object, json_schema and text"
                            + " are supported)",
                    type);
            if ("json_schema".equals(type)) {
                // OpenAI's shape: {type, json_schema: {name, schema, strict?}}. The schema is the
                // whole point of the request, so a missing or non-object one is refused here
                // rather than silently degrading to unconstrained text.
                Map<String, Object> wrapper =
                        Values.asObject(m.get("json_schema"), "response_format.json_schema");
                Values.asObject(wrapper.get("schema"), "response_format.json_schema.schema");
            }
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
     * A field the request actually carries: an explicit null means "unset", as in OpenAI's SDKs.
     */
    private static boolean present(Map<String, Object> request, String field) {
        return request.get(field) != null;
    }

    /**
     * Sampling-parameter validation shared by all endpoints; called on the HTTP handler thread
     * (before queueing, and before any SSE headers) so invalid requests fail fast with a 400.
     */
    static void validateGenerationParams(Map<String, Object> request, ServerConfig config) {
        // OPTIONAL, because this server has exactly one model: an absent (or blank) "model" is
        // unambiguous - it can only mean the served one, which is what Requests.modelId already
        // returned. Naming the WRONG model is still a real mistake and still refused. Requiring
        // the field bought no safety and cost every curl and every client that omits it a 400.
        if (request.containsKey("model") && request.get("model") != null) {
            require(request.get("model") instanceof String, "model must be a string");
            String name = (String) request.get("model");
            require(
                    name.isBlank() || name.equalsIgnoreCase(config.modelName()),
                    "Unknown model: %s (this server serves %s)",
                    name,
                    config.modelName());
        }
        if ((request.containsKey("grammar") || request.containsKey("response_format"))
                && !config.limits().grammar()) {
            require(false, "Grammar constraints disabled (--no-grammar)");
        }
        require(Values.intValue(request.get("n"), 1) == 1, "Only n=1 is supported");
        // Every range below is checked ONLY when the request carries the field. These are rules
        // about what a CLIENT may ask for (OpenAI's caps); the server's own defaults are the
        // operator's business and are validated where they are built - Sampling's constructor and
        // Options. Checking the RESOLVED value instead meant a server started with `--temp 9`
        // answered "Invalid argument: temperature must be within [0, 2]" to every request that did
        // not override it: the client blamed for the operator's flag, and the same shape of bug
        // that made a stock `--server` refuse every request omitting max_tokens.
        if (present(request, "temperature")) {
            float temperature = Values.floatValue(request.get("temperature"), 1f);
            require(
                    Float.isFinite(temperature) && 0 <= temperature && temperature <= 2,
                    "Invalid argument: temperature must be within [0, 2]");
        }
        if (present(request, "top_p")) {
            float topp = Values.floatValue(request.get("top_p"), 1f);
            require(
                    Float.isFinite(topp) && 0 <= topp && topp <= 1,
                    "Invalid argument: top_p must be within [0, 1]");
        }
        if (present(request, "top_k")) {
            require(
                    Values.intValue(request.get("top_k"), 0) >= 0,
                    "Invalid argument: top_k must be non-negative (0 disables it)");
        }
        if (present(request, "min_p")) {
            float minp = Values.floatValue(request.get("min_p"), 0f);
            require(
                    Float.isFinite(minp) && 0 <= minp && minp <= 1,
                    "Invalid argument: min_p must be within [0, 1]");
        }
        if (Requests.budget(request) != null) {
            require(
                    -1 <= Values.intValue(Requests.budget(request), -1),
                    "Invalid argument: max_tokens must be -1 (context-bounded) or non-negative");
        }
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
