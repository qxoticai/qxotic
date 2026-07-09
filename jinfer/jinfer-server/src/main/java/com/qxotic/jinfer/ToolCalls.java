// Tool-call parsing: turns a model reply into normalized OpenAI tool_calls. Recognizes the
// three shapes LFM2.5 emits — native <|tool_call_start|>...<|tool_call_end|> blocks, a JSON
// tool-call envelope, and bare Pythonic [name(args)] text — independent of HTTP/transport.
package com.qxotic.jinfer;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

final class ToolCalls {

    private ToolCalls() {}

    static final String TC_START = "<|tool_call_start|>";
    static final String TC_END = "<|tool_call_end|>";

    /**
     * Parse tool calls from a model reply, trying the three shapes LFM2.5 is known to emit, in
     * descending order of confidence: native {@code <|tool_call_start|>...<|tool_call_end|>}
     * blocks, a JSON tool-call envelope, then bare Pythonic {@code [name(args)]} text.
     */
    static List<Map<String, Object>> parseToolCalls(String text, Set<String> knownTools) {
        List<Map<String, Object>> nativeCalls = parseNativeToolCalls(text, knownTools);
        if (!nativeCalls.isEmpty()) return nativeCalls;
        String stripped = text.strip(); // both fallbacks work on the trimmed reply; strip once
        List<Map<String, Object>> jsonCalls = parseJsonToolCalls(stripped);
        if (!jsonCalls.isEmpty()) return jsonCalls;
        return parseBarePythonic(stripped, knownTools);
    }

    /** The OpenAI streaming-delta form of parsed calls: each call with its {@code index}. */
    static List<Map<String, Object>> toolCallDeltas(List<Map<String, Object>> toolCalls) {
        List<Map<String, Object>> deltas = new ArrayList<>();
        for (int i = 0; i < toolCalls.size(); i++) {
            Map<String, Object> call = toolCalls.get(i);
            Map<String, Object> delta = new LinkedHashMap<>(call);
            delta.put("index", i);
            deltas.add(delta);
        }
        return deltas;
    }

    /**
     * All {@code <|tool_call_start|>...<|tool_call_end|>} blocks parsed per the format LFM2.5 was
     * trained on (reference: SGLang Lfm2Detector): each block holds either a Pythonic call list
     * {@code [f(a=1), g(b='x')]} (or a single bare call) or a JSON array/object of {@code {name,
     * arguments}}. Calls naming a function absent from {@code knownTools} are dropped (a non-empty
     * set validates; empty = accept all).
     */
    private static List<Map<String, Object>> parseNativeToolCalls(
            String text, Set<String> knownTools) {
        List<Map<String, Object>> calls = new ArrayList<>();
        int pos = 0;
        while (true) {
            int start = text.indexOf(TC_START, pos);
            if (start < 0) break;
            int end = text.indexOf(TC_END, start + TC_START.length());
            if (end < 0) break; // truncated mid-call: nothing reliable to parse
            String content = text.substring(start + TC_START.length(), end).strip();
            pos = end + TC_END.length();
            for (Map<String, Object> call : parseToolCallBlock(content)) {
                if (!knownTools.isEmpty()
                        && !knownTools.contains(Values.stringValue(call.get("name"), ""))) {
                    System.err.println(
                            "dropping tool call to undefined function: " + call.get("name"));
                    continue;
                }
                Map<String, Object> normalized = normalizeToolCall(call, calls.size());
                if (normalized != null) calls.add(normalized);
            }
        }
        return calls;
    }

    /**
     * One block's content as raw {@code {name, arguments}} maps: JSON-looking content parses as
     * JSON, anything else as the Pythonic form; malformed content yields no calls.
     */
    private static List<Map<String, Object>> parseToolCallBlock(String content) {
        if (content.startsWith("{") || content.startsWith("[{")) {
            try {
                Object parsed = JsonCodec.parse(content);
                List<?> list = parsed instanceof List<?> l ? l : List.of(parsed);
                List<Map<String, Object>> calls = new ArrayList<>();
                for (Object value : list) calls.add(Values.asObject(value, "tool call"));
                return calls;
            } catch (RuntimeException e) {
                // fall through: '{' can also open a Pythonic dict literal
            }
        }
        try {
            return new PythonicCalls(content).parse();
        } catch (RuntimeException e) {
            System.err.println("unparseable tool call block: " + e.getMessage());
            return List.of();
        }
    }

    /**
     * Lenient JSON fallback: an OpenAI-style {@code {"tool_calls":[...]}} envelope, a single {@code
     * {"function_call":{...}}} or {@code {"name":.., "arguments":..}} object, or a bare JSON array
     * of call objects — extracted from anywhere in the (already-stripped) text, fenced code
     * included. Yields no calls when the text holds no parseable JSON of a recognized shape.
     */
    private static List<Map<String, Object>> parseJsonToolCalls(String stripped) {
        String json = extractJson(stripped);
        if (json.isEmpty()) return List.of();
        try {
            List<?> calls = jsonCallList(JsonCodec.parse(json));
            if (calls == null) return List.of();
            List<Map<String, Object>> out = new ArrayList<>();
            for (Object value : calls) {
                Map<String, Object> normalized =
                        normalizeToolCall(Values.asObject(value, "tool call"), out.size());
                if (normalized != null) out.add(normalized);
            }
            return out;
        } catch (RuntimeException e) {
            return List.of(); // malformed JSON or an unexpected element shape
        }
    }

    /** The raw call objects carried by a recognized JSON tool-call shape, or null if none. */
    private static List<?> jsonCallList(Object parsed) {
        if (parsed instanceof List<?> list) return list;
        if (parsed instanceof Map<?, ?> map) {
            if (map.get("tool_calls") instanceof List<?> list) return list;
            if (map.get("function_call") instanceof Map<?, ?> call) return List.of(call);
            if (map.get("name") instanceof String
                    && (map.containsKey("arguments") || map.containsKey("parameters"))) {
                return List.of(map);
            }
        }
        return null;
    }

    /**
     * Fallback: scan the whole text for a bare pythonic tool call — a bracketed list {@code
     * [name(args), ...]} or a single {@code name(args)} — emitted without the {@code
     * <|tool_call_start|>} markers (a documented LFM2.5 behavior, see llama.cpp #24178). The scan
     * tries to parse a call sequence at every plausible start offset; matching is fully
     * string-aware (it uses {@link PythonicCalls}, not bracket counting), so quoted argument values
     * may contain brackets, parens, or commas without breaking detection. Requires a non-empty
     * {@code knownTools}, and accepts a run only when every call names a known tool — this keeps
     * ordinary prose ({@code [text](url)}, {@code print()}) from being mistaken for a tool call.
     */
    private static List<Map<String, Object>> parseBarePythonic(
            String text, Set<String> knownTools) {
        if (text.isEmpty() || knownTools.isEmpty()) return List.of();
        for (int p = 0; p < text.length(); p++) {
            char c = text.charAt(p);
            // A call sequence starts either with '[' (list form) or with the first character of
            // an identifier at a word boundary (bare-call form, e.g. `get_weather(...)`).
            boolean listStart = c == '[';
            boolean callStart =
                    (Character.isLetter(c) || c == '_')
                            && (p == 0 || !isIdentifierPart(text.charAt(p - 1)));
            if (!listStart && !callStart) continue;
            List<Map<String, Object>> calls = tryParseCallsAt(text, p, knownTools);
            if (calls != null) return calls;
        }
        return List.of();
    }

    /**
     * Attempt to parse a call sequence starting exactly at {@code from}. Returns the normalized
     * calls when parsing succeeds, the run is non-empty, and every name is a known tool; otherwise
     * {@code null} (so the caller keeps scanning).
     */
    private static List<Map<String, Object>> tryParseCallsAt(
            String text, int from, Set<String> knownTools) {
        PythonicCalls parser = new PythonicCalls(text);
        parser.i = from;
        List<Map<String, Object>> calls;
        try {
            calls = parser.parseCallSequence();
        } catch (RuntimeException e) {
            return null; // not a well-formed call sequence at this offset
        }
        if (calls.isEmpty()) return null;
        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> call : calls) {
            if (!knownTools.contains(Values.stringValue(call.get("name"), ""))) return null;
            Map<String, Object> normalized = normalizeToolCall(call, out.size());
            if (normalized != null) out.add(normalized);
        }
        return out.isEmpty() ? null : out;
    }

    /** Whether {@code c} can appear inside a (dotted) function identifier. */
    private static boolean isIdentifierPart(char c) {
        return Character.isLetterOrDigit(c) || c == '_' || c == '.';
    }

    private static String extractJson(String text) {
        if (text.startsWith("```")) {
            int firstNewline = text.indexOf('\n');
            int lastFence = text.lastIndexOf("```");
            if (firstNewline >= 0 && lastFence > firstNewline)
                return text.substring(firstNewline + 1, lastFence).strip();
        }
        int objectStart = text.indexOf('{');
        int arrayStart = text.indexOf('[');
        int start =
                objectStart < 0
                        ? arrayStart
                        : arrayStart < 0 ? objectStart : Math.min(objectStart, arrayStart);
        if (start < 0) return "";
        int end = Math.max(text.lastIndexOf('}'), text.lastIndexOf(']'));
        return end >= start ? text.substring(start, end + 1).strip() : "";
    }

    private static Map<String, Object> normalizeToolCall(Map<String, Object> call, int index) {
        Object functionValue = call.get("function");
        String name;
        Object arguments;
        if (functionValue instanceof Map<?, ?> function) {
            name = Values.stringValue(function.get("name"), null);
            arguments = function.get("arguments");
        } else {
            name = Values.stringValue(call.get("name"), null);
            arguments = call.get("arguments");
        }
        if (name == null || name.isBlank()) return null;
        String argumentString =
                arguments instanceof String s
                        ? s
                        : JsonCodec.stringify(arguments == null ? Map.of() : arguments);
        Map<String, Object> function = new LinkedHashMap<>();
        function.put("name", name);
        function.put("arguments", argumentString);
        Map<String, Object> normalized = new LinkedHashMap<>();
        normalized.put(
                "id",
                Values.stringValue(
                        call.get("id"),
                        "call_" + Long.toUnsignedString(System.nanoTime(), 36) + "_" + index));
        normalized.put("type", "function");
        normalized.put("function", function);
        return normalized;
    }

    /**
     * Recursive-descent parser for the Pythonic tool-call syntax: {@code [name(k=v, ...), ...]} or
     * a single bare call; values are Python literals — strings (either quote, backslash escapes),
     * numbers, True/False/None, and nested lists/tuples/dicts — converted to their JSON-ready Java
     * shapes. Positional arguments are skipped (matching SGLang); anything else malformed throws.
     */
    private static final class PythonicCalls {
        private final String s;
        private int i;

        PythonicCalls(String s) {
            this.s = s;
        }

        /**
         * Parse a call sequence — either a bracketed list {@code [f(..), g(..)]} or a single bare
         * call {@code f(..)} — starting at the current offset, leaving {@code i} just past the
         * closing bracket (or the call). String contents are honored, so brackets, parens, or
         * commas inside quoted argument values never terminate the scan early.
         */
        List<Map<String, Object>> parseCallSequence() {
            List<Map<String, Object>> calls = new ArrayList<>();
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

        /** Parse the entire input as exactly one call sequence; trailing junk is an error. */
        List<Map<String, Object>> parse() {
            List<Map<String, Object>> calls = parseCallSequence();
            skipWs();
            if (i < s.length()) throw err("end of input");
            return calls;
        }

        private Map<String, Object> call() {
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
                        literal(); // positional argument: parse and skip (SGLang behavior)
                    }
                    skipWs();
                    char c = next();
                    if (c == ')') break;
                    if (c != ',') throw err("',' or ')'");
                }
            }
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("name", name);
            call.put("arguments", arguments);
            return call;
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
                                default -> esc; // \' \" \\ and anything exotic pass through
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
            if (floating) return Double.parseDouble(token);
            return Long.parseLong(token);
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
                } // trailing comma (and 1-tuples)
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
