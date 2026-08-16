package com.qxotic.jinfer.chat;

import com.qxotic.format.json.Json;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses the payload inside a claimed tool-call span into structured {@link Content.ToolCall}s. Two
 * grammars, because the families emit two: a JSON object/array of {@code {name, arguments}}
 * (Granite, SmolLM3) and the XML function form {@code <function=NAME><parameter=K>...} shared by
 * Qwen 3.5 and Nemotron. (The pythonic call list is LFM2's own; see its codec.)
 *
 * <p>Only the payload text is parsed here; the span boundaries are the reply language's job and are
 * decided on token ids, so this never has to guard against content faking a marker. A payload that
 * parses as neither grammar yields an empty list - the walk drops the call without ending the
 * reply.
 */
public final class ToolCallSyntax {

    private ToolCallSyntax() {}

    /**
     * Parse one span's content as a JSON call envelope: a single {@code {name, arguments}} object
     * or an array of them. {@code id} on each call is blank - the caller assigns one if the wire
     * needs it.
     */
    public static List<Content.ToolCall> parseBlock(String content) {
        String c = content.strip();
        if (c.isEmpty() || !(c.startsWith("{") || c.startsWith("[{") || c.startsWith("[ {"))) {
            return List.of();
        }
        try {
            // JsonCodec, not raw Json: argument types must be the engine's value model
            // (double decimals, Java null) on EVERY family's wire
            return fromJson(parseLenient(c));
        } catch (RuntimeException notJson) {
            return List.of();
        }
    }

    /**
     * Strict first, then one salvage pass with trailing commas removed: small models emit {@code
     * {"city": "Paris",}} often enough that langchain4j's own tool layer strips them too - a
     * strict-only parse DROPS the whole call here, silently.
     */
    private static Object parseLenient(String text) {
        try {
            return JsonCodec.parse(text);
        } catch (RuntimeException strict) {
            return JsonCodec.parse(stripTrailingCommas(text));
        }
    }

    /** Commas immediately before a closing brace/bracket, removed outside string literals. */
    static String stripTrailingCommas(String json) {
        StringBuilder out = new StringBuilder(json.length());
        boolean inString = false, escaped = false;
        for (int i = 0; i < json.length(); i++) {
            char ch = json.charAt(i);
            if (inString) {
                out.append(ch);
                if (escaped) escaped = false;
                else if (ch == '\\') escaped = true;
                else if (ch == '"') inString = false;
            } else if (ch == '"') {
                inString = true;
                out.append(ch);
            } else if (ch == ',') {
                int next = i + 1;
                while (next < json.length() && Character.isWhitespace(json.charAt(next))) next++;
                if (next < json.length()
                        && (json.charAt(next) == '}' || json.charAt(next) == ']')) {
                    continue; // trailing comma: drop
                }
                out.append(ch);
            } else {
                out.append(ch);
            }
        }
        return out.toString();
    }

    /**
     * Parse one XML-function span, {@code <function=NAME><parameter=K>\nV\n</parameter>...
     * </function>} - the form Qwen 3.5 and Nemotron emit between their trusted {@code <tool_call>}
     * / {@code </tool_call>} ids (one function per span; both templates emit one span per call). A
     * parameter value is the template's {@code tojson}-for-objects / raw-string-otherwise, so it
     * parses as JSON when it is valid JSON (numbers, objects, arrays, booleans) and stays a plain
     * string otherwise (an unquoted word like {@code Paris} is not valid JSON).
     */
    public static List<Content.ToolCall> parseFunctionXml(String block) {
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
            // the templates frame the value as ">\n" + value + "\n</parameter>"
            String value = body.substring(keyEnd + 2, close);
            if (!key.isEmpty()) arguments.put(key, typedValue(value));
            p = body.indexOf("<parameter=", close);
        }
        return List.of(new Content.ToolCall("", name, arguments));
    }

    /**
     * A parameter value as its JSON type when it is valid JSON, else the raw string - with the
     * Python spellings the templates PRINT ({@code x | string} renders {@code True}/{@code
     * False}/{@code None}) typed back, so booleans round-trip instead of arriving as "True".
     */
    private static Object typedValue(String value) {
        switch (value) {
            case "True" -> {
                return Boolean.TRUE;
            }
            case "False" -> {
                return Boolean.FALSE;
            }
            case "None" -> {
                return null;
            }
            default -> {}
        }
        try {
            return JsonCodec.parse(value);
        } catch (RuntimeException notJson) {
            return value;
        }
    }

    /**
     * A JSON object payload as the engine's value model (Mistral's {@code [ARGS]} body), or null
     * when the text does not parse as a JSON object.
     */
    public static Map<String, Object> parseObject(String text) {
        try {
            if (parseLenient(text) instanceof Map<?, ?> parsed) {
                Map<String, Object> out = new LinkedHashMap<>();
                for (Map.Entry<?, ?> e : parsed.entrySet()) {
                    out.put(String.valueOf(e.getKey()), e.getValue());
                }
                return out;
            }
        } catch (RuntimeException malformed) {
            // not a JSON object: no call
        }
        return null;
    }

    /**
     * The tool's inner {@code function} object from its definition envelope ({@code {"type":
     * "function","function":{...}}}), lenient when the definition IS already that object.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> functionObject(Tool tool) {
        Map<String, Object> definition = tool.definition();
        return definition.get("function") instanceof Map<?, ?> fn
                ? (Map<String, Object>) fn
                : definition;
    }

    /**
     * A template value as Jinja renders it: {@code tojson} for maps and lists, {@code |string}
     * otherwise (Python spellings: {@code True}/{@code False}/{@code None}).
     */
    public static String jinjaValue(Object v) {
        if (v instanceof Map || v instanceof List) return jinjaJson(v);
        if (v instanceof Boolean b) return b ? "True" : "False";
        if (v == null) return "None";
        return String.valueOf(v);
    }

    /**
     * Serialize a parsed JSON value exactly as Jinja's {@code tojson} filter does: {@code ", "} and
     * {@code ": "} separators, map insertion order preserved, JSON booleans/null (lowercase). This
     * is the canonical form a model was trained on, so both the tool-definition JSON and
     * object-valued call arguments must go through it to stay byte-exact with the model's template.
     */
    public static String jinjaJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeJinja(sb, value);
        return sb.toString();
    }

    private static void writeJinja(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("null");
        } else if (v instanceof String s) {
            sb.append(Json.stringify(s));
        } else if (v instanceof Boolean b) {
            sb.append(b.booleanValue());
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(Json.stringify(String.valueOf(e.getKey()))).append(": ");
                writeJinja(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) sb.append(", ");
                writeJinja(sb, l.get(i));
            }
            sb.append(']');
        } else {
            sb.append(v); // numbers
        }
    }

    /** A JSON tool-call payload: a single {@code {name,arguments}} object or an array of them. */
    private static List<Content.ToolCall> fromJson(Object parsed) {
        List<?> list = parsed instanceof List<?> l ? l : List.of(parsed);
        List<Content.ToolCall> calls = new ArrayList<>();
        for (Object value : list) {
            if (!(value instanceof Map<?, ?> m)) continue;
            Object name = m.get("name");
            if (!(name instanceof String n) || n.isEmpty()) continue;
            Object args = m.containsKey("arguments") ? m.get("arguments") : m.get("parameters");
            calls.add(new Content.ToolCall("", n, asArguments(args)));
        }
        return calls;
    }

    /** Coerce a JSON arguments value (object, or a JSON string holding an object) to a map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asArguments(Object args) {
        if (args instanceof Map<?, ?> m) return (Map<String, Object>) m;
        if (args instanceof String s && !s.isBlank()) {
            try {
                if (JsonCodec.parse(s) instanceof Map<?, ?> m) return (Map<String, Object>) m;
            } catch (RuntimeException notJson) {
                // a plain string argument value - keep it under a conventional key
            }
            return Map.of("value", s);
        }
        return Map.of();
    }
}
