package com.qxotic.jinfer.models.lfm2;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.chat.Content;
import com.qxotic.jinfer.chat.Tool;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** LFM 2.5's Python-style tool wire and Jinja-compatible JSON rendering. */
final class Lfm2ToolCodec {
    private static final Json.ParseOptions JSON_OPTIONS =
            Json.ParseOptions.defaults().decimalsAsBigDecimal(false);

    private Lfm2ToolCodec() {}

    static List<Content.ToolCall> parse(String payload) {
        String text = payload.strip();
        if (text.isEmpty()) return List.of();
        if (text.startsWith("{")
                || (text.startsWith("[") && text.substring(1).stripLeading().startsWith("{"))) {
            try {
                return jsonCalls(json(text));
            } catch (RuntimeException ignored) {
                // A Python dictionary also starts with '{'.
            }
        }
        try {
            return new Python(text).parse();
        } catch (RuntimeException ignored) {
            return List.of();
        }
    }

    /**
     * The template's {@code render_tool_calls}: {@code [name(arg=value, ...)]}. {@code escape}
     * selects the 2.6B {@code format_arg_value} (escaped strings, JSON lists) over the 8B one (raw
     * strings, Python repr lists); mappings are JSON in both.
     */
    static String renderCalls(List<Content.ToolCall> calls, boolean escape) {
        StringBuilder out = new StringBuilder("[");
        for (int i = 0; i < calls.size(); i++) {
            if (i > 0) out.append(", ");
            Content.ToolCall call = calls.get(i);
            out.append(call.name()).append('(');
            boolean first = true;
            for (Map.Entry<String, Object> argument : call.arguments().entrySet()) {
                if (!first) out.append(", ");
                first = false;
                out.append(argument.getKey()).append('=');
                writeArgument(out, argument.getValue(), escape);
            }
            out.append(')');
        }
        return out.append(']').toString();
    }

    static String renderTools(List<Tool> tools) {
        StringBuilder out = new StringBuilder("List of tools: [");
        for (int i = 0; i < tools.size(); i++) {
            if (i > 0) out.append(", ");
            writeJson(out, tools.get(i).definition());
        }
        return out.append(']').toString();
    }

    private static void writeArgument(StringBuilder out, Object value, boolean escape) {
        if (value instanceof String text && !escape) {
            out.append('\'').append(text).append('\'');
        } else if (value instanceof List<?> list && !escape) {
            writePython(out, list);
        } else if (value instanceof String text) {
            out.append('\'');
            for (int i = 0; i < text.length(); i++) {
                char c = text.charAt(i);
                switch (c) {
                    case '\\' -> out.append("\\\\");
                    case '\'' -> out.append("\\'");
                    case '\n' -> out.append("\\n");
                    case '\r' -> out.append("\\r");
                    default -> out.append(c);
                }
            }
            out.append('\'');
        } else if (value instanceof Map<?, ?> || value instanceof List<?>) {
            writeJson(out, value);
        } else if (value instanceof Boolean bool) {
            out.append(bool ? "True" : "False");
        } else if (value == null) {
            out.append("None");
        } else {
            out.append(value);
        }
    }

    /** Python {@code str()} of a value: what {@code | string} renders for an 8B list argument. */
    private static void writePython(StringBuilder out, Object value) {
        switch (value) {
            case null -> out.append("None");
            case String text -> {
                // repr: single quotes, backslash escapes (a string holding a quote and no
                // double quote would be double-quoted by Python; ponytail, kept single)
                out.append('\'');
                for (int i = 0; i < text.length(); i++) {
                    char c = text.charAt(i);
                    switch (c) {
                        case '\\' -> out.append("\\\\");
                        case '\'' -> out.append("\\'");
                        case '\n' -> out.append("\\n");
                        case '\r' -> out.append("\\r");
                        case '\t' -> out.append("\\t");
                        default -> out.append(c);
                    }
                }
                out.append('\'');
            }
            case Boolean bool -> out.append(bool ? "True" : "False");
            case List<?> list -> {
                out.append('[');
                for (int i = 0; i < list.size(); i++) {
                    if (i > 0) out.append(", ");
                    writePython(out, list.get(i));
                }
                out.append(']');
            }
            case Map<?, ?> map -> {
                out.append('{');
                boolean first = true;
                for (Map.Entry<?, ?> entry : map.entrySet()) {
                    if (!first) out.append(", ");
                    first = false;
                    writePython(out, String.valueOf(entry.getKey()));
                    out.append(": ");
                    writePython(out, entry.getValue());
                }
                out.append('}');
            }
            default -> out.append(value);
        }
    }

    private static void writeJson(StringBuilder out, Object value) {
        if (value == null) {
            out.append("null");
        } else if (value instanceof String text) {
            out.append(Json.stringify(text));
        } else if (value instanceof Boolean || value instanceof Number) {
            out.append(value);
        } else if (value instanceof Map<?, ?> map) {
            out.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) out.append(", ");
                first = false;
                out.append(Json.stringify(String.valueOf(entry.getKey()))).append(": ");
                writeJson(out, entry.getValue());
            }
            out.append('}');
        } else if (value instanceof List<?> list) {
            out.append('[');
            for (int i = 0; i < list.size(); i++) {
                if (i > 0) out.append(", ");
                writeJson(out, list.get(i));
            }
            out.append(']');
        } else {
            throw new IllegalArgumentException("not a JSON value: " + value.getClass().getName());
        }
    }

    private static Object json(String text) {
        return fromJsonValue(Json.parse(text, JSON_OPTIONS));
    }

    private static Object fromJsonValue(Object value) {
        if (value == Json.NULL) return null;
        if (value instanceof Map<?, ?> map) {
            LinkedHashMap<String, Object> out = new LinkedHashMap<>();
            map.forEach((key, item) -> out.put(String.valueOf(key), fromJsonValue(item)));
            return out;
        }
        if (value instanceof List<?> list) {
            return list.stream().map(Lfm2ToolCodec::fromJsonValue).toList();
        }
        return value;
    }

    private static List<Content.ToolCall> jsonCalls(Object value) {
        List<?> values = value instanceof List<?> list ? list : List.of(value);
        List<Content.ToolCall> calls = new ArrayList<>();
        for (Object candidate : values) {
            if (!(candidate instanceof Map<?, ?> outer)) continue;
            Object body = outer.get("function");
            Map<?, ?> call = body instanceof Map<?, ?> function ? function : outer;
            if (!(call.get("name") instanceof String name) || name.isEmpty()) continue;
            Object arguments =
                    call.containsKey("arguments") ? call.get("arguments") : call.get("parameters");
            if (arguments instanceof String text) {
                try {
                    arguments = json(text);
                } catch (RuntimeException ignored) {
                    arguments = Map.of("value", text);
                }
            }
            @SuppressWarnings("unchecked")
            Map<String, Object> map =
                    arguments instanceof Map<?, ?> object ? (Map<String, Object>) object : Map.of();
            calls.add(new Content.ToolCall("", name, map));
        }
        return calls;
    }

    private static final class Python {
        private final String text;
        private int at;

        Python(String text) {
            this.text = text;
        }

        List<Content.ToolCall> parse() {
            List<Content.ToolCall> calls = calls();
            whitespace();
            if (at != text.length()) throw error("end of input");
            return calls;
        }

        private List<Content.ToolCall> calls() {
            List<Content.ToolCall> out = new ArrayList<>();
            whitespace();
            if (peek() != '[') {
                out.add(call());
                return out;
            }
            at++;
            whitespace();
            if (peek() == ']') {
                at++;
                return out;
            }
            while (true) {
                out.add(call());
                whitespace();
                char next = next();
                if (next == ']') return out;
                if (next != ',') throw error("',' or ']'");
            }
        }

        private Content.ToolCall call() {
            String name = identifier();
            whitespace();
            if (next() != '(') throw error("'('");
            Map<String, Object> arguments = new LinkedHashMap<>();
            whitespace();
            if (peek() == ')') {
                at++;
                return new Content.ToolCall("", name, arguments);
            }
            while (true) {
                whitespace();
                int mark = at;
                String key = identifier();
                whitespace();
                if (peek() == '=') {
                    at++;
                    arguments.put(key, literal());
                } else {
                    at = mark;
                    literal();
                }
                whitespace();
                char next = next();
                if (next == ')') return new Content.ToolCall("", name, arguments);
                if (next != ',') throw error("',' or ')'");
            }
        }

        private Object literal() {
            whitespace();
            char next = peek();
            if (next == '\'' || next == '"') return string();
            if (next == '[') return sequence('[', ']');
            if (next == '(') return sequence('(', ')');
            if (next == '{') return dictionary();
            if (next == '+' || next == '-' || next == '.' || Character.isDigit(next))
                return number();
            return switch (identifier()) {
                case "True", "true" -> Boolean.TRUE;
                case "False", "false" -> Boolean.FALSE;
                case "None", "null" -> null;
                default -> throw error("literal");
            };
        }

        private String string() {
            char quote = next();
            StringBuilder out = new StringBuilder();
            while (at < text.length()) {
                char next = text.charAt(at++);
                if (next == quote) {
                    if (closesString()) return out.toString();
                    out.append(next);
                } else if (next == '\\' && at < text.length()) {
                    char escaped = text.charAt(at++);
                    out.append(
                            switch (escaped) {
                                case 'n' -> '\n';
                                case 'r' -> '\r';
                                case 't' -> '\t';
                                case '0' -> '\0';
                                default -> escaped;
                            });
                } else {
                    out.append(next);
                }
            }
            throw error("closing quote");
        }

        private boolean closesString() {
            int next = at;
            while (next < text.length() && Character.isWhitespace(text.charAt(next))) next++;
            return next == text.length() || ",)]}:".indexOf(text.charAt(next)) >= 0;
        }

        private Number number() {
            int from = at;
            if (peek() == '+' || peek() == '-') at++;
            boolean decimal = false;
            while (at < text.length()) {
                char next = text.charAt(at);
                if (Character.isDigit(next)) {
                    at++;
                } else if (next == '.' || next == 'e' || next == 'E') {
                    decimal = true;
                    at++;
                } else if ((next == '+' || next == '-')
                        && at > from
                        && (text.charAt(at - 1) == 'e' || text.charAt(at - 1) == 'E')) {
                    at++;
                } else {
                    break;
                }
            }
            String value = text.substring(from, at);
            if (decimal) return Double.parseDouble(value);
            return Long.parseLong(value);
        }

        private List<Object> sequence(char open, char close) {
            if (next() != open) throw error("'" + open + "'");
            List<Object> out = new ArrayList<>();
            whitespace();
            if (peek() == close) {
                at++;
                return out;
            }
            while (true) {
                out.add(literal());
                whitespace();
                char next = next();
                if (next == close) return out;
                if (next != ',') throw error("',' or '" + close + "'");
                whitespace();
                if (peek() == close) {
                    at++;
                    return out;
                }
            }
        }

        private Map<String, Object> dictionary() {
            if (next() != '{') throw error("'{'");
            Map<String, Object> out = new LinkedHashMap<>();
            whitespace();
            if (peek() == '}') {
                at++;
                return out;
            }
            while (true) {
                Object key = literal();
                whitespace();
                if (next() != ':') throw error("':'");
                out.put(String.valueOf(key), literal());
                whitespace();
                char next = next();
                if (next == '}') return out;
                if (next != ',') throw error("',' or '}'");
                whitespace();
                if (peek() == '}') {
                    at++;
                    return out;
                }
            }
        }

        private String identifier() {
            whitespace();
            int from = at;
            while (at < text.length()) {
                char next = text.charAt(at);
                if (!Character.isLetterOrDigit(next) && next != '_' && next != '.') break;
                at++;
            }
            if (from == at) throw error("identifier");
            return text.substring(from, at);
        }

        private void whitespace() {
            while (at < text.length() && Character.isWhitespace(text.charAt(at))) at++;
        }

        private char peek() {
            return at < text.length() ? text.charAt(at) : '\0';
        }

        private char next() {
            if (at == text.length()) throw error("more input");
            return text.charAt(at++);
        }

        private IllegalArgumentException error(String expected) {
            return new IllegalArgumentException(
                    "expected " + expected + " at offset " + at + " in: " + text);
        }
    }
}
