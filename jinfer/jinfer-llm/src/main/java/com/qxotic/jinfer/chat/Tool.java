package com.qxotic.jinfer.chat;

/**
 * A tool the model may call: its name, a natural-language description, and a JSON-Schema object
 * describing its parameters. Conversation-scoped - a {@link TurnTemplate} renders the whole tool
 * list into the preamble (the system/developer turn), where every cacheable template puts it, so
 * the prompt prefix stays turn-stable across the conversation.
 *
 * <p>{@code parametersJsonSchema} is kept as the raw schema string (the OpenAI {@code
 * function.parameters} object verbatim); the template embeds it, and only the wire layer parses or
 * produces JSON.
 */
public record Tool(String name, String description, String parametersJsonSchema) {
    public Tool {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty tool name");
        description = description == null ? "" : description;
        parametersJsonSchema = parametersJsonSchema == null ? "{}" : parametersJsonSchema;
    }
}
