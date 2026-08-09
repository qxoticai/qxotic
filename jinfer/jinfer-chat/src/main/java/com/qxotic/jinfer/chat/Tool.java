package com.qxotic.jinfer.chat;

/**
 * A tool the model may call: its {@code name} (identity), and {@code rawJson} - the whole tool
 * object exactly as the request supplied it (typically {@code
 * {"type":"function","function":{...}}}). Conversation-scoped: a {@link TurnTemplate} renders the
 * tool list into the preamble (the system/developer turn), where every cacheable template puts it,
 * so the prompt prefix stays turn-stable across the conversation.
 *
 * <p>The raw JSON is verbatim on purpose. Templates serialize the tool with Jinja {@code tojson};
 * re-serializing from decomposed fields would not preserve key order or spacing and would break
 * byte-exactness with the model's oracle template. The decode side (the call detector and its
 * allow-list) keys on {@code name}; the encode side embeds {@code rawJson}.
 */
public record Tool(String name, String rawJson) {
    public Tool {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty tool name");
        if (rawJson == null || rawJson.isEmpty())
            throw new IllegalArgumentException("empty tool rawJson");
    }

    /**
     * The tool's JSON-Schema parameter object, for schema-bound argument grammars (the reply
     * language's forced calls). Reads {@code parameters} from the raw function object, unwrapping a
     * {@code {"type":"function","function":{...}}} envelope; a tool declaring none gets the empty
     * object schema - its forced arguments admit exactly {@code {}}.
     */
    @SuppressWarnings("unchecked")
    public java.util.Map<String, Object> parameters() {
        Object node = JsonCodec.parse(rawJson);
        if (node instanceof java.util.Map<?, ?> m
                && m.get("function") instanceof java.util.Map<?, ?> fn) {
            node = fn;
        }
        if (node instanceof java.util.Map<?, ?> m
                && m.get("parameters") instanceof java.util.Map<?, ?> p) {
            return (java.util.Map<String, Object>) p;
        }
        return java.util.Map.of("type", "object");
    }
}
