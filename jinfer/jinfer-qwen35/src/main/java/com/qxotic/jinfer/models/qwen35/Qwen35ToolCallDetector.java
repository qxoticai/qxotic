package com.qxotic.jinfer.models.qwen35;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.chat.Part;
import com.qxotic.jinfer.chat.ToolCallDetector;
import com.qxotic.jinfer.llm.GgufTokenizer;
import java.io.ByteArrayOutputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Qwen3.5 tool-call detector. A call is the span between the trusted {@code <tool_call>} and {@code
 * </tool_call>} token ids; its payload is Qwen's XML function form, {@code <function=NAME>
 * <parameter=KEY>\nVALUE\n</parameter>...</function>}. Each span holds exactly one function (the
 * template emits one {@code <tool_call>} per call). Because the boundary is matched on ids, not
 * decoded text, conversation content can never fake a call.
 *
 * <p>A parameter value is Qwen's {@code tojson}-for-objects / raw-{@code string}-otherwise, so it
 * is parsed as JSON when it is valid JSON (numbers, objects, arrays, booleans) and kept as a plain
 * string otherwise (an unquoted word like {@code Paris} is not valid JSON and stays a string).
 */
public final class Qwen35ToolCallDetector implements ToolCallDetector {

    private final GgufTokenizer tokenizer;
    private final int startMarker;
    private final int endMarker;

    private boolean inSpan;
    private final ByteArrayOutputStream span = new ByteArrayOutputStream();
    private final List<Part.ToolCall> calls = new ArrayList<>();

    public Qwen35ToolCallDetector(GgufTokenizer tokenizer) {
        this.tokenizer = tokenizer;
        this.startMarker = tokenizer.requiredSpecial("<tool_call>");
        this.endMarker = tokenizer.requiredSpecial("</tool_call>");
    }

    @Override
    public boolean accept(int token) {
        if (token == startMarker) {
            inSpan = true;
            span.reset();
            return true;
        }
        if (token == endMarker) {
            if (inSpan) {
                parse(span.toString(StandardCharsets.UTF_8));
                span.reset();
                inSpan = false;
            }
            return true;
        }
        if (inSpan) {
            byte[] bytes = tokenizer.decodeTokenBytes(token);
            span.write(bytes, 0, bytes.length);
            return true;
        }
        return false;
    }

    /** Parse one {@code <function=NAME>...<parameter=K>\nV\n</parameter>...</function>} span. */
    private void parse(String block) {
        int fn = block.indexOf("<function=");
        if (fn < 0) return;
        int nameEnd = block.indexOf('>', fn);
        if (nameEnd < 0) return;
        String name = block.substring(fn + "<function=".length(), nameEnd).strip();
        if (name.isEmpty()) return;

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
        calls.add(new Part.ToolCall("", name, arguments));
    }

    /** A parameter value as its JSON type when it is valid JSON, else the raw string. */
    private static Object typed(String value) {
        try {
            return Json.parse(value);
        } catch (RuntimeException notJson) {
            return value;
        }
    }

    @Override
    public List<Part.ToolCall> calls() {
        return List.copyOf(calls);
    }
}
