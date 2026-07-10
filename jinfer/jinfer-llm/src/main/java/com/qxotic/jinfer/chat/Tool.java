package com.qxotic.jinfer.chat;

/**
 * A tool the model may call: its name, a natural-language description, and a JSON-Schema object
 * describing its parameters. Conversation-scoped - a {@link TurnTemplate} renders the whole tool
 * list into the preamble (the system/developer turn), where every cacheable template puts it, so
 * the prompt prefix stays turn-stable across the conversation.
 *
 * <p>Two JSON views, on purpose. {@code parametersJsonSchema} is the parameters object alone, for
 * introspection. {@code rawJson} is the WHOLE tool object exactly as the request supplied it
 * (typically {@code {"type":"function","function":{...}}}) - templates that serialize the tool with
 * Jinja {@code tojson} must embed this verbatim, because re-serializing from the decomposed fields
 * would not preserve key order or spacing and would break byte-exactness with the model's oracle
 * template. The decode side (detector, {@code knownTools}) uses {@code name}; the encode side uses
 * {@code rawJson}.
 */
public record Tool(String name, String description, String parametersJsonSchema, String rawJson) {
    public Tool {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty tool name");
        description = description == null ? "" : description;
        parametersJsonSchema = parametersJsonSchema == null ? "{}" : parametersJsonSchema;
        if (rawJson == null || rawJson.isEmpty())
            throw new IllegalArgumentException("empty tool rawJson");
    }
}
