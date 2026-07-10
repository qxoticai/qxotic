package com.qxotic.jinfer.chat;

import com.qxotic.format.json.Json;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses the payload inside a tool-call span into structured {@link Part.ToolCall}s. Two grammars,
 * because the models emit two: a JSON object/array of {@code {name, arguments}}, or the Pythonic
 * call list {@code [f(a=1), g(b='x')]} that LFM2.5 (and other pythonic-tool-call models) produce -
 * a format the wider ecosystem special-cases too (SGLang's {@code Lfm2Detector}, llama.cpp).
 *
 * <p>Only the payload text is parsed here; the span boundaries are the model detector's job and are
 * decided on token ids, so this never has to guard against content faking a marker. Shared by every
 * model's {@link ToolCallDetector} and by the server's whole-render fallback.
 */
public final class ToolCallSyntax {

    private ToolCallSyntax() {}

    /**
     * Parse one span's content, trying JSON first (a {@code {...}} or {@code [{...}]} payload) and
     * falling back to the Pythonic grammar. Returns an empty list when the content parses as
     * neither. {@code id} on each call is blank - the caller assigns one if the wire needs it.
     */
    public static List<Part.ToolCall> parseBlock(String content) {
        String c = content.strip();
        if (c.isEmpty()) return List.of();
        if (c.startsWith("{") || c.startsWith("[{") || c.startsWith("[ {")) {
            try {
                return fromJson(Json.parse(c));
            } catch (RuntimeException notJson) {
                // '{' also opens a Pythonic dict literal - fall through to the pythonic parser
            }
        }
        try {
            return new Pythonic(c).parse();
        } catch (RuntimeException notPythonic) {
            return List.of();
        }
    }

    /** A JSON tool-call payload: a single {@code {name,arguments}} object or an array of them. */
    private static List<Part.ToolCall> fromJson(Object parsed) {
        List<?> list = parsed instanceof List<?> l ? l : List.of(parsed);
        List<Part.ToolCall> calls = new ArrayList<>();
        for (Object value : list) {
            if (!(value instanceof Map<?, ?> m)) continue;
            Object name = m.get("name");
            if (!(name instanceof String n) || n.isEmpty()) continue;
            Object args = m.containsKey("arguments") ? m.get("arguments") : m.get("parameters");
            calls.add(new Part.ToolCall("", n, asArguments(args)));
        }
        return calls;
    }

    /** Coerce a JSON arguments value (object, or a JSON string holding an object) to a map. */
    @SuppressWarnings("unchecked")
    private static Map<String, Object> asArguments(Object args) {
        if (args instanceof Map<?, ?> m) return (Map<String, Object>) m;
        if (args instanceof String s && !s.isBlank()) {
            try {
                if (Json.parse(s) instanceof Map<?, ?> m) return (Map<String, Object>) m;
            } catch (RuntimeException notJson) {
                // a plain string argument value - keep it under a conventional key
            }
            return Map.of("value", s);
        }
        return Map.of();
    }

    /**
     * Recursive-descent parser for the pythonic call grammar: a bracketed list {@code [f(..),
     * g(..)]} or a single bare call {@code f(..)}. Fully string-aware, so brackets, parens or
     * commas inside quoted argument values never end a span early. Positional arguments are parsed
     * and dropped (the tool schema is keyword-only), matching SGLang.
     */
    private static final class Pythonic {
        private final String s;
        private int i;

        Pythonic(String s) {
            this.s = s;
        }

        List<Part.ToolCall> parse() {
            List<Part.ToolCall> calls = parseCallSequence();
            skipWs();
            if (i < s.length()) throw err("end of input");
            return calls;
        }

        private List<Part.ToolCall> parseCallSequence() {
            List<Part.ToolCall> calls = new ArrayList<>();
            skipWs();
            if (peek() == '[') {
                i++;
                skipWs();
                if (peek() == ']') {
                    i++;
                    return calls;
                }
                while (true) {
                    calls.add(call());
                    skipWs();
                    char c = next();
                    if (c == ']') break;
                    if (c != ',') throw err("',' or ']'");
                }
            } else {
                calls.add(call());
            }
            return calls;
        }

        private Part.ToolCall call() {
            skipWs();
            String name = identifier();
            skipWs();
            if (next() != '(') throw err("'('");
            Map<String, Object> arguments = new LinkedHashMap<>();
            skipWs();
            if (peek() == ')') {
                i++;
            } else {
                while (true) {
                    skipWs();
                    int mark = i;
                    String key = identifier();
                    skipWs();
                    if (peek() == '=') {
                        i++;
                        arguments.put(key, literal());
                    } else {
                        i = mark;
                        literal(); // positional argument: parse and skip
                    }
                    skipWs();
                    char c = next();
                    if (c == ')') break;
                    if (c != ',') throw err("',' or ')'");
                }
            }
            return new Part.ToolCall("", name, arguments);
        }

        private Object literal() {
            skipWs();
            char c = peek();
            if (c == '"' || c == '\'') return string();
            if (c == '[') return sequence('[', ']');
            if (c == '(') return sequence('(', ')'); // tuple -> JSON array
            if (c == '{') return dict();
            if (c == '-' || c == '+' || Character.isDigit(c) || c == '.') return number();
            String word = identifier();
            return switch (word) {
                case "True", "true" -> Boolean.TRUE;
                case "False", "false" -> Boolean.FALSE;
                case "None", "null" -> null;
                default -> throw err("literal");
            };
        }

        private String string() {
            char quote = next();
            StringBuilder out = new StringBuilder();
            while (true) {
                if (i >= s.length()) throw err("closing quote");
                char c = s.charAt(i++);
                if (c == quote) return out.toString();
                if (c == '\\' && i < s.length()) {
                    char esc = s.charAt(i++);
                    out.append(
                            switch (esc) {
                                case 'n' -> '\n';
                                case 't' -> '\t';
                                case 'r' -> '\r';
                                case '0' -> '\0';
                                default -> esc;
                            });
                } else {
                    out.append(c);
                }
            }
        }

        private Object number() {
            int from = i;
            if (peek() == '-' || peek() == '+') i++;
            boolean floating = false;
            while (i < s.length()) {
                char c = s.charAt(i);
                if (Character.isDigit(c)) i++;
                else if (c == '.' || c == 'e' || c == 'E') {
                    floating = true;
                    i++;
                } else if ((c == '-' || c == '+')
                        && (s.charAt(i - 1) == 'e' || s.charAt(i - 1) == 'E')) i++;
                else break;
            }
            String token = s.substring(from, i);
            return floating ? Double.parseDouble(token) : Long.parseLong(token);
        }

        private List<Object> sequence(char open, char close) {
            if (next() != open) throw err("'" + open + "'");
            List<Object> out = new ArrayList<>();
            skipWs();
            if (peek() == close) {
                i++;
                return out;
            }
            while (true) {
                out.add(literal());
                skipWs();
                char c = next();
                if (c == close) return out;
                if (c != ',') throw err("',' or '" + close + "'");
                skipWs();
                if (peek() == close) {
                    i++;
                    return out;
                }
            }
        }

        private Map<String, Object> dict() {
            if (next() != '{') throw err("'{'");
            Map<String, Object> out = new LinkedHashMap<>();
            skipWs();
            if (peek() == '}') {
                i++;
                return out;
            }
            while (true) {
                Object key = literal();
                skipWs();
                if (next() != ':') throw err("':'");
                out.put(String.valueOf(key), literal());
                skipWs();
                char c = next();
                if (c == '}') return out;
                if (c != ',') throw err("',' or '}'");
                skipWs();
                if (peek() == '}') {
                    i++;
                    return out;
                }
            }
        }

        private String identifier() {
            skipWs();
            int from = i;
            while (i < s.length() && isIdentifierPart(s.charAt(i))) i++;
            if (i == from) throw err("identifier");
            return s.substring(from, i);
        }

        private static boolean isIdentifierPart(char c) {
            return Character.isLetterOrDigit(c) || c == '_' || c == '.';
        }

        private void skipWs() {
            while (i < s.length() && Character.isWhitespace(s.charAt(i))) i++;
        }

        private char peek() {
            return i < s.length() ? s.charAt(i) : '\0';
        }

        private char next() {
            if (i >= s.length()) throw err("more input");
            return s.charAt(i++);
        }

        private IllegalArgumentException err(String expected) {
            return new IllegalArgumentException(
                    "expected " + expected + " at offset " + i + " in: " + s);
        }
    }
}
