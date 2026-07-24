package com.qxotic.jinfer.chat;

import com.qxotic.format.json.Json;
import com.qxotic.jinfer.llm.Grammar;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Parses the payload inside a tool-call span into structured {@link Part.ToolCall}s. Three
 * grammars, because the models emit three: a JSON object/array of {@code {name, arguments}}, the
 * Pythonic call list {@code [f(a=1), g(b='x')]} that LFM2.5 (and other pythonic-tool-call models)
 * produce - a format the wider ecosystem special-cases too (SGLang's {@code Lfm2Detector},
 * llama.cpp) - and the XML function form {@code <function=NAME><parameter=K>...} shared by Qwen 3.5
 * and Nemotron.
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

    /**
     * Render calls as the pythonic list body {@code name(k=v, ...), ...} (WITHOUT the surrounding
     * brackets or markers - the template adds those). The inverse of {@link #parseBlock}'s pythonic
     * branch, and the exact shape LFM2's {@code render_tool_calls} macro emits: string values are
     * single-quoted, object values are JSON, everything else is its Python {@code str} (so booleans
     * are {@code True}/{@code False} and {@code null} is {@code None}). Byte-exactness with a
     * model's template is verified by that model's oracle test.
     */
    public static String renderPythonic(List<Part.ToolCall> calls) {
        StringBuilder sb = new StringBuilder();
        for (int c = 0; c < calls.size(); c++) {
            if (c > 0) sb.append(", ");
            Part.ToolCall call = calls.get(c);
            sb.append(call.name()).append('(');
            boolean first = true;
            for (Map.Entry<String, Object> arg : call.arguments().entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                sb.append(arg.getKey()).append('=').append(formatArgValue(arg.getValue()));
            }
            sb.append(')');
        }
        return sb.toString();
    }

    private static String formatArgValue(Object value) {
        if (value instanceof String s) return "'" + s + "'";
        if (value instanceof Map || value instanceof List) return jinjaJson(value);
        if (value instanceof Boolean b) return b ? "True" : "False";
        if (value == null) return "None";
        return String.valueOf(value);
    }

    /**
     * Serialize a parsed JSON value exactly as Jinja's {@code tojson} filter does: {@code ", "} and
     * {@code ": "} separators, map insertion order preserved, JSON booleans/null (lowercase). This
     * is the canonical form a model was trained on, so both the tool-definition JSON (which the
     * server canonicalizes from the request) and object-valued call arguments must go through it to
     * stay byte-exact with the model's template.
     */
    public static String jinjaJson(Object value) {
        StringBuilder sb = new StringBuilder();
        writeJinja(sb, value);
        return sb.toString();
    }

    /**
     * The tool's inner {@code function} object from its raw JSON envelope ({@code {"type":
     * "function","function":{...}}}), lenient when the raw JSON IS already that object - the one
     * wire fact about {@link Tool#rawJson} every template port needs.
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> functionObject(Tool tool) {
        Map<String, Object> raw = (Map<String, Object>) JsonCodec.parse(tool.rawJson());
        return raw.get("function") instanceof Map<?, ?> fn ? (Map<String, Object>) fn : raw;
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
     * Python {@code str()} of a parsed JSON value - {@code {'key': 'val', 'n': 1}} with
     * single-quoted strings and {@code True}/{@code False}/{@code None} - the form templates that
     * render tools with {@code tool | string} trained their models on (SmolLM3).
     */
    public static String pythonRepr(Object v) {
        StringBuilder sb = new StringBuilder();
        writePython(sb, v);
        return sb.toString();
    }

    private static void writePython(StringBuilder sb, Object v) {
        if (v == null) {
            sb.append("None");
        } else if (v instanceof String s) {
            sb.append('\'').append(s.replace("\\", "\\\\").replace("'", "\\'")).append('\'');
        } else if (v instanceof Boolean b) {
            sb.append(b ? "True" : "False");
        } else if (v instanceof Map<?, ?> m) {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!first) sb.append(", ");
                first = false;
                writePython(sb, String.valueOf(e.getKey()));
                sb.append(": ");
                writePython(sb, e.getValue());
            }
            sb.append('}');
        } else if (v instanceof List<?> l) {
            sb.append('[');
            for (int i = 0; i < l.size(); i++) {
                if (i > 0) sb.append(", ");
                writePython(sb, l.get(i));
            }
            sb.append(']');
        } else {
            sb.append(v); // numbers
        }
    }

    /**
     * Parse one XML-function span, {@code <function=NAME><parameter=K>\nV\n</parameter>...
     * </function>} - the form Qwen 3.5 and Nemotron emit between their trusted {@code <tool_call>}
     * / {@code </tool_call>} ids (one function per span; both templates emit one span per call). A
     * parameter value is the template's {@code tojson}-for-objects / raw-string-otherwise, so it
     * parses as JSON when it is valid JSON (numbers, objects, arrays, booleans) and stays a plain
     * string otherwise (an unquoted word like {@code Paris} is not valid JSON).
     */
    public static List<Part.ToolCall> parseFunctionXml(String block) {
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
        return List.of(new Part.ToolCall("", name, arguments));
    }

    /** A parameter value as its JSON type when it is valid JSON, else the raw string. */
    private static Object typedValue(String value) {
        try {
            return JsonCodec.parse(value);
        } catch (RuntimeException notJson) {
            return value;
        }
    }

    /**
     * A prefix-pin GBNF over the offered tool names - {@code prefix (name|...|name)} - the shared
     * shape behind {@link ChatTemplate#callGrammar}: every family's call syntax opens with
     * plain-byte framing then the name, and pinning exactly that much guarantees a call of an
     * offered tool while leaving everything after the name free.
     *
     * <p>The pin deliberately ENDS AT THE NAME: pinning the delimiter too forces an unnatural token
     * split (the model's training merges the delimiter with the first argument - {@code (city}),
     * and generation derails at the off-distribution boundary (hallucinated arguments, observed on
     * LFM2.5). Same boundary lesson as the server's forced-call seeding, which seeds {@code [name}
     * and never the paren. Ceiling: a tool name that is a strict prefix of another offered name
     * resolves toward the longer one.
     */
    public static String prefixPinGbnf(String prefix, List<Tool> tools) {
        StringBuilder names = new StringBuilder();
        for (Tool t : tools) {
            if (!names.isEmpty()) names.append(" | ");
            names.append(Grammar.gbnfLiteral(t.name()));
        }
        return "root ::= " + Grammar.gbnfLiteral(prefix) + " (" + names + ")";
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
                if (c == quote) {
                    if (closesString()) return out.toString();
                    out.append(c); // lenient: an unescaped inner quote is content (models emit
                    continue; //     them verbatim despite the syntax)
                }
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

        /**
         * A quote CLOSES the string only when what follows can continue the grammar; anything else
         * means the model emitted an unescaped quote inside the value. Ceiling: content containing
         * quote-then-delimiter (e.g. {@code "hi", she said}) still closes early - the whole parse
         * then fails and the span is no call, exactly as before this leniency.
         */
        private boolean closesString() {
            int j = i;
            while (j < s.length() && Character.isWhitespace(s.charAt(j))) j++;
            return j >= s.length() || ",)]}:".indexOf(s.charAt(j)) >= 0;
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
            // two statements on purpose: a ?: would numeric-promote the long branch to double,
            // turning every integer argument into 2.0
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
