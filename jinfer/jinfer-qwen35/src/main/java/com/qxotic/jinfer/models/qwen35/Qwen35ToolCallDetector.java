package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.chat.Part;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Qwen3.5 tool-call payload parsing: the payload between the trusted {@code <tool_call>} and {@code
 * </tool_call>} ids (the span itself is claimed by the generic {@link
 * com.qxotic.jinfer.chat.SpanToolCallDetector}) is Qwen's XML function form, {@code <function=NAME>
 * <parameter=KEY>\nVALUE\n</parameter>...</function>}. Each span holds exactly one function (the
 * template emits one {@code <tool_call>} per call).
 *
 * <p>A parameter value is Qwen's {@code tojson}-for-objects / raw-{@code string}-otherwise, so it
 * is parsed as JSON when it is valid JSON (numbers, objects, arrays, booleans) and kept as a plain
 * string otherwise (an unquoted word like {@code Paris} is not valid JSON and stays a string).
 */
public final class Qwen35ToolCallDetector {

    private Qwen35ToolCallDetector() {}

    /** Parse one {@code <function=NAME>...<parameter=K>\nV\n</parameter>...</function>} span. */
    static List<Part.ToolCall> parsePayload(String block) {
        int fn = block.indexOf("<function=");
        if (fn < 0) return List.of();
        int nameEnd = block.indexOf('>', fn);
        if (nameEnd < 0) return List.of();
        String name = block.substring(fn + "<function=".length(), nameEnd).strip();
        if (name.isEmpty()) return List.of();

        int fnClose = block.indexOf("</function>", nameEnd);
        String body = block.substring(nameEnd, fnClose < 0 ? block.length() : fnClose);

        Map<String, Object> arguments = new LinkedHashMap<>();
        int p = body.indexOf("<parameter=");
        while (p >= 0) {
            int keyEnd = body.indexOf('>', p);
            if (keyEnd < 0) break;
            String key = body.substring(p + "<parameter=".length(), keyEnd).strip();
            int close = body.indexOf("\n</parameter>", keyEnd);
            if (close < 0) break;
            // template frames the value as ">\n" + value + "\n</parameter>"
            String value = body.substring(keyEnd + 2, close);
            if (!key.isEmpty()) arguments.put(key, typed(value));
            p = body.indexOf("<parameter=", close);
        }
        return List.of(new Part.ToolCall("", name, arguments));
    }

    /** A parameter value as its JSON type when it is valid JSON, else the raw string. */
    private static Object typed(String value) {
        try {
            return Json.parse(value);
        } catch (RuntimeException notJson) {
            return value;
        }
    }
}
