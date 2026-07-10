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
 * byte-exactness with the model's oracle template. The decode side (detector, {@code knownTools})
 * keys on {@code name}; the encode side embeds {@code rawJson}.
 */
public record Tool(String name, String rawJson) {
    public Tool {
        if (name == null || name.isEmpty()) throw new IllegalArgumentException("empty tool name");
        if (rawJson == null || rawJson.isEmpty())
            throw new IllegalArgumentException("empty tool rawJson");
    }
}
