package com.qxotic.jinfer.models.gemma4;

import com.qxotic.jinfer.chat.Part;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Gemma 4's compact tool notation, a byte-exact port of the GGUF template's macros
 * (format_function_declaration / format_parameters / format_argument): {@code key:value} pairs with
 * string values delimited by the QUOTE TOKEN {@code <|"|>}, dictsorted keys, uppercased schema
 * types. Rendering emits through a {@link Sink} - plain text runs and quote marks separately - so
 * the template can emit the quote as a TRUSTED id while every text segment stays plain-encoded
 * (tool descriptions, arguments, and results can never mint control tokens).
 *
 * <p>The parse side reads a claimed {@code <|tool_call>} span's decoded text, where the quote token
 * appears as its literal {@code <|"|>} rendering.
 */
final class Gemma4ToolSyntax {

    private Gemma4ToolSyntax() {}

    /** Rendering target: text runs plain-encode, quotes emit the trusted {@code <|"|>} id. */
    interface Sink {
        void text(String s);

        void quote();
    }

    private static final List<String> STANDARD_KEYS =
            List.of("description", "type", "properties", "required", "nullable");

    /** {@code format_function_declaration}: one {@code <|tool>...<tool|>} block's payload. */
    @SuppressWarnings("unchecked")
    static void declaration(Map<String, Object> tool, Sink out) {
        Map<String, Object> fn = (Map<String, Object>) tool.getOrDefault("function", tool);
        out.text("declaration:" + fn.getOrDefault("name", "") + "{description:");
        quoted(String.valueOf(fn.getOrDefault("description", "")), out);
        Object paramsRaw = fn.get("parameters");
        if (truthy(paramsRaw) && paramsRaw instanceof Map<?, ?> params) {
            out.text(",parameters:{");
            Object props = params.get("properties");
            if (truthy(props)) {
                out.text("properties:{");
                formatParameters((Map<String, Object>) props, out, false);
                out.text("},");
            }
            Object required = params.get("required");
            if (truthy(required)) {
                out.text("required:[");
                quotedList((List<Object>) required, out);
                out.text("],");
            }
            Object type = params.get("type");
            if (truthy(type)) {
                out.text("type:");
                quoted(String.valueOf(type).toUpperCase(), out);
                out.text("}");
            }
        }
        Object response = fn.get("response");
        if (response instanceof Map<?, ?> resp) {
            out.text(",response:{");
            Object desc = resp.get("description");
            if (truthy(desc)) {
                out.text("description:");
                quoted(String.valueOf(desc), out);
                out.text(",");
            }
            if ("OBJECT".equals(String.valueOf(resp.get("type")).toUpperCase())) {
                out.text("type:");
                quoted("OBJECT", out);
                out.text("}");
            }
        }
        out.text("}");
    }

    /**
     * {@code format_parameters}: dictsorted properties, each {@code key:{...,type:<|"|>T<|"|>}}.
     */
    @SuppressWarnings("unchecked")
    private static void formatParameters(Map<String, Object> properties, Sink out, boolean filter) {
        boolean first = true;
        for (Map.Entry<String, Object> e : dictsort(properties)) {
            if (filter && STANDARD_KEYS.contains(e.getKey())) continue;
            if (!first) out.text(",");
            first = false;
            Map<String, Object> value =
                    e.getValue() instanceof Map<?, ?> m ? (Map<String, Object>) m : Map.of();
            out.text(e.getKey() + ":{");
            boolean comma = false;
            Object desc = value.get("description");
            if (truthy(desc)) {
                out.text("description:");
                quoted(String.valueOf(desc), out);
                comma = true;
            }
            String type = String.valueOf(value.getOrDefault("type", "")).toUpperCase();
            if ("STRING".equals(type) && truthy(value.get("enum"))) {
                if (comma) out.text(",");
                comma = true;
                out.text("enum:");
                formatArgument(value.get("enum"), true, out);
            } else if ("ARRAY".equals(type)
                    && value.get("items") instanceof Map<?, ?> items
                    && !items.isEmpty()) {
                if (comma) out.text(",");
                comma = true;
                out.text("items:{");
                boolean firstItem = true;
                for (Map.Entry<String, Object> item : dictsort((Map<String, Object>) items)) {
                    if (item.getValue() == null) continue;
                    if (!firstItem) out.text(",");
                    firstItem = false;
                    switch (item.getKey()) {
                        case "properties" -> {
                            out.text("properties:{");
                            if (item.getValue() instanceof Map<?, ?> p) {
                                formatParameters((Map<String, Object>) p, out, false);
                            }
                            out.text("}");
                        }
                        case "required" -> {
                            out.text("required:[");
                            quotedList((List<Object>) item.getValue(), out);
                            out.text("]");
                        }
                        case "type" -> {
                            out.text("type:");
                            if (item.getValue() instanceof String s) {
                                formatArgument(s.toUpperCase(), true, out);
                            } else if (item.getValue() instanceof List<?> types) {
                                List<Object> upper = new ArrayList<>();
                                for (Object t : types) upper.add(String.valueOf(t).toUpperCase());
                                formatArgument(upper, true, out);
                            }
                        }
                        default -> {
                            out.text(item.getKey() + ":");
                            formatArgument(item.getValue(), true, out);
                        }
                    }
                }
                out.text("}");
            }
            if (truthy(value.get("nullable"))) {
                if (comma) out.text(",");
                comma = true;
                out.text("nullable:true");
            }
            if ("OBJECT".equals(type)) {
                Object props = value.get("properties");
                if (props instanceof Map<?, ?> p) {
                    if (comma) out.text(",");
                    comma = true;
                    out.text("properties:{");
                    formatParameters((Map<String, Object>) p, out, false);
                    out.text("}");
                } else {
                    if (comma) out.text(",");
                    comma = true;
                    out.text("properties:{");
                    formatParameters(value, out, true);
                    out.text("}");
                }
                if (truthy(value.get("required"))) {
                    if (comma) out.text(",");
                    comma = true;
                    out.text("required:[");
                    quotedList((List<Object>) value.get("required"), out);
                    out.text("]");
                }
            }
            if (comma) out.text(",");
            out.text("type:");
            quoted(type, out);
            out.text("}");
        }
    }

    /** A call turn's span payload: {@code call:name{k:v,...}}, args dictsorted, keys unquoted. */
    static void call(String name, Map<String, Object> args, Sink out) {
        out.text("call:" + name + "{");
        boolean first = true;
        for (Map.Entry<String, Object> e : dictsort(args)) {
            if (!first) out.text(",");
            first = false;
            out.text(e.getKey() + ":");
            formatArgument(e.getValue(), false, out);
        }
        out.text("}");
    }

    /** A response block's payload: {@code response:name{value:<|"|>text<|"|>}} (string results). */
    static void response(String name, String text, Sink out) {
        out.text("response:" + name + "{value:");
        formatArgument(text, false, out);
        out.text("}");
    }

    /** {@code format_argument}: strings quoted with the quote token, maps/lists recursed. */
    @SuppressWarnings("unchecked")
    private static void formatArgument(Object v, boolean escapeKeys, Sink out) {
        switch (v) {
            case String s -> quoted(s, out);
            case Boolean b -> out.text(b ? "true" : "false");
            case Map<?, ?> m -> {
                out.text("{");
                boolean first = true;
                for (Map.Entry<String, Object> e : dictsort((Map<String, Object>) m)) {
                    if (!first) out.text(",");
                    first = false;
                    if (escapeKeys) quoted(e.getKey(), out);
                    else out.text(e.getKey());
                    out.text(":");
                    formatArgument(e.getValue(), escapeKeys, out);
                }
                out.text("}");
            }
            case List<?> list -> {
                out.text("[");
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) out.text(",");
                    formatArgument(list.get(i), escapeKeys, out);
                }
                out.text("]");
            }
            case null -> out.text("None");
            default -> out.text(String.valueOf(v)); // numbers render as Jinja does (3, 3.0)
        }
    }

    private static void quoted(String s, Sink out) {
        out.quote();
        out.text(s);
        out.quote();
    }

    private static void quotedList(List<Object> items, Sink out) {
        for (int i = 0; i < items.size(); i++) {
            if (i > 0) out.text(",");
            quoted(String.valueOf(items.get(i)), out);
        }
    }

    /** Jinja dictsort: entries ordered by key, case-insensitively, stable. */
    private static List<Map.Entry<String, Object>> dictsort(Map<String, Object> map) {
        List<Map.Entry<String, Object>> entries = new ArrayList<>(map.entrySet());
        entries.sort(Map.Entry.comparingByKey(String.CASE_INSENSITIVE_ORDER));
        return entries;
    }

    private static boolean truthy(Object v) {
        return switch (v) {
            case null -> false;
            case String s -> !s.isEmpty();
            case Map<?, ?> m -> !m.isEmpty();
            case List<?> l -> !l.isEmpty();
            case Boolean b -> b;
            default -> true;
        };
    }

    // ---- parse side: a claimed <|tool_call> span's decoded text -----------------

    /** The quote token's literal rendering inside decoded span text. */
    static final String QUOTE = "<|\"|>";

    /**
     * Parses {@code call:name{k:v,...}} into one call; malformed payloads parse to no calls (the
     * span stays visible as text - honest failure over a guessed call).
     */
    static List<Part.ToolCall> parseBlock(String payload) {
        try {
            Cursor c = new Cursor(payload.strip());
            c.expect("call:");
            String name = c.until('{').strip();
            Map<String, Object> args = c.parseMap();
            if (name.isEmpty()) return List.of();
            return List.of(new Part.ToolCall("", name, args));
        } catch (RuntimeException malformed) {
            return List.of();
        }
    }

    private static final class Cursor {
        private final String s;
        private int at;

        Cursor(String s) {
            this.s = s;
        }

        void expect(String prefix) {
            if (!s.startsWith(prefix, at)) throw new IllegalStateException("expected " + prefix);
            at += prefix.length();
        }

        String until(char stop) {
            int i = s.indexOf(stop, at);
            if (i < 0) throw new IllegalStateException("missing " + stop);
            String out = s.substring(at, i);
            at = i;
            return out;
        }

        void ws() {
            while (at < s.length() && Character.isWhitespace(s.charAt(at))) at++;
        }

        Map<String, Object> parseMap() {
            expect("{");
            Map<String, Object> out = new LinkedHashMap<>();
            ws();
            if (s.startsWith("}", at)) {
                at++;
                return out;
            }
            while (true) {
                ws();
                String key = until(':').strip();
                at++; // ':'
                ws();
                out.put(key, parseValue());
                ws();
                if (s.startsWith(",", at)) {
                    at++;
                    continue;
                }
                expect("}");
                return out;
            }
        }

        Object parseValue() {
            if (s.startsWith(QUOTE, at)) {
                at += QUOTE.length();
                int end = s.indexOf(QUOTE, at);
                if (end < 0) throw new IllegalStateException("unterminated string");
                String out = s.substring(at, end);
                at = end + QUOTE.length();
                return out;
            }
            if (s.startsWith("{", at)) return parseMap();
            if (s.startsWith("[", at)) {
                at++;
                List<Object> out = new ArrayList<>();
                ws();
                if (s.startsWith("]", at)) {
                    at++;
                    return out;
                }
                while (true) {
                    ws();
                    out.add(parseValue());
                    ws();
                    if (s.startsWith(",", at)) {
                        at++;
                        continue;
                    }
                    expect("]");
                    return out;
                }
            }
            int end = at;
            while (end < s.length() && ",}]".indexOf(s.charAt(end)) < 0) end++;
            String token = s.substring(at, end).strip();
            at = end;
            return switch (token) {
                case "true" -> Boolean.TRUE;
                case "false" -> Boolean.FALSE;
                case "None", "null" -> null;
                default -> {
                    try {
                        yield Long.valueOf(token);
                    } catch (NumberFormatException notLong) {
                        yield Double.valueOf(token); // malformed numbers throw -> no call parsed
                    }
                }
            };
        }
    }
}
