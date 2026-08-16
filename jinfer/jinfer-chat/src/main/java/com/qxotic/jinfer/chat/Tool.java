package com.qxotic.jinfer.chat;

import java.util.Map;

/** A callable tool and its complete JSON-shaped definition. */
public record Tool(String name, Map<String, Object> definition) {
    public Tool {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty tool name");
        definition = JsonValues.object(definition);
    }

    /** The function's parameter schema, or the schema for an empty object when omitted. */
    @SuppressWarnings("unchecked")
    public Map<String, Object> parameters() {
        Object function = definition.get("function");
        Map<String, Object> body =
                function instanceof Map<?, ?> map ? (Map<String, Object>) map : definition;
        Object parameters = body.get("parameters");
        return parameters instanceof Map<?, ?> map
                ? (Map<String, Object>) map
                : Map.of("type", "object");
    }
}
