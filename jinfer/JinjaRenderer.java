package com.llama4j;

import com.qxotic.format.json.Json;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Minimal Jinja template renderer for LLM chat templates.
 * Two-phase architecture (lexer → parser, like llama.cpp's jinja/):
 * tokenize the template into a flat token stream, then parse the stream
 * into an AST, then execute the AST against a variable context.
 */
public final class JinjaRenderer {

    // ── Value system ──────────────────────────────────────────────

    sealed interface Val {
        default boolean truthy() { return true; }
        default String asStr() { return toString(); }
        default boolean isString() { return this instanceof Str; }
        default boolean isMapping() { return this instanceof Obj; }
        default boolean isSequence() { return this instanceof Arr; }
        default boolean isDefined() { return !(this instanceof Undef); }
        default boolean isNone() { return this instanceof None; }

        record Int(long v) implements Val {
            @Override public boolean truthy() { return v != 0; }
            @Override public String asStr() { return Long.toString(v); }
        }
        record Flt(double v) implements Val {
            @Override public boolean truthy() { return v != 0.0; }
            @Override public String asStr() { return fmtDouble(v); }
        }
        record Str(String v) implements Val {
            @Override public boolean truthy() { return !v.isEmpty(); }
            @Override public String asStr() { return v; }
        }
        record Bool(boolean v) implements Val {
            @Override public boolean truthy() { return v; }
            @Override public String asStr() { return v ? "True" : "False"; }
        }
        record Arr(List<Val> v) implements Val {
            @Override public boolean truthy() { return !v.isEmpty(); }
            // Python/Jinja str(list): elements use repr (strings quoted)
            @Override public String asStr() {
                var sb = new StringBuilder("[");
                for (int i = 0; i < v.size(); i++) {
                    if (i > 0) sb.append(", ");
                    sb.append(pyRepr(v.get(i)));
                }
                return sb.append("]").toString();
            }
        }
        record Obj(LinkedHashMap<String,Val> v, boolean hasBuiltins) implements Val {
            Obj(LinkedHashMap<String,Val> v) { this(v, true); }
            @Override public boolean truthy() { return !v.isEmpty(); }
            // Python/Jinja str(dict): {'key': <repr>, ...} — what models trained on `tools | string`
            @Override public String asStr() {
                var sb = new StringBuilder("{");
                boolean first = true;
                for (var e : v.entrySet()) {
                    if (!first) sb.append(", ");
                    first = false;
                    sb.append('\'').append(e.getKey()).append("': ").append(pyRepr(e.getValue()));
                }
                return sb.append("}").toString();
            }
            Val get(String key) {
                Val found = v.get(key);
                if (found != null) return found;
                if (!hasBuiltins) return new Undef(key);
                Val.Obj self = this;
                return switch (key) {
                    case "items"  -> new Val.Func("items", args -> {
                        var arr = new ArrayList<Val>();
                        for (var e : self.v.entrySet())
                            arr.add(new Val.Arr(List.of(new Val.Str(e.getKey()), e.getValue())));
                        return new Val.Arr(arr);
                    });
                    case "keys"   -> new Val.Func("keys", args -> {
                        var keys = new ArrayList<Val>();
                        for (String k : self.v.keySet()) keys.add(new Val.Str(k));
                        return new Val.Arr(keys);
                    });
                    case "values" -> new Val.Func("values", args -> {
                        return new Val.Arr(new ArrayList<>(self.v.values()));
                    });
                    case "get"    -> new Val.Func("get", args -> {
                        String k = expectStr(requireArg(args, 0));
                        Val def = args.size() > 1 ? requireArg(args, 1) : Val.NONE;
                        Val f = self.v.get(k);
                        return f != null ? f : def;
                    });
                    default       -> new Undef(key);
                };
            }
            void set(String key, Val val) { v.put(key, val); }
            boolean has(String key) { return v.containsKey(key); }
        }
        record Func(String name, Callable fn) implements Val {
            @Override public String asStr() { return "[function " + name + "]"; }
            @FunctionalInterface interface Callable { Val call(List<Val> args); }
        }
        record None() implements Val {
            @Override public boolean truthy() { return false; }
            @Override public String asStr() { return "None"; }
        }
        record Undef(String hint) implements Val {
            @Override public boolean truthy() { return false; }
            @Override public String asStr() { return ""; }
        }

        static final None NONE = new None();

        static Val of(Object o) {
            return switch (o) {
                case null          -> NONE;
                case Val v         -> v;
                case Boolean b     -> new Bool(b);
                case Integer i     -> new Int(i);
                case Long l        -> new Int(l);
                case Double d      -> new Flt(d);
                case Float f       -> new Flt(f);
                case Number n      -> new Flt(n.doubleValue());
                case String s      -> new Str(s);
                case Map<?,?> m -> {
                    var obj = new LinkedHashMap<String,Val>();
                    for (var e : m.entrySet())
                        obj.put(String.valueOf(e.getKey()), of(e.getValue()));
                    yield new Obj(obj);
                }
                case List<?> l -> {
                    var arr = new ArrayList<Val>(l.size());
                    for (var item : l) arr.add(of(item));
                    yield new Arr(arr);
                }
                case Iterable<?> it -> {
                    var arr = new ArrayList<Val>();
                    for (var item : it) arr.add(of(item));
                    yield new Arr(arr);
                }
                default -> new Str(String.valueOf(o));
            };
        }
    }

    private static String fmtDouble(double v) {
        if (v == (long) v) return Long.toString((long) v);
        String s = Double.toString(v);
        if (s.endsWith(".0")) s = s.substring(0, s.length() - 2);
        return s;
    }

    /** Python {@code repr()} of a value as used inside a list/dict's {@code str()} — strings are
     *  single-quoted, True/False/None capitalized, containers recurse. (Plain top-level
     *  {@code str()} does NOT quote strings; that's {@link Val#asStr()}.) */
    static String pyRepr(Val v) {
        return switch (v) {
            case Val.Str s -> "'" + s.v.replace("\\", "\\\\").replace("'", "\\'") + "'";
            case Val.Bool b -> b.v ? "True" : "False";
            case Val.None n -> "None";
            default -> v.asStr(); // numbers, and Arr/Obj which already render as reprs
        };
    }

    /** Thrown when a template uses a construct this minimal engine does not implement. Failing
     *  loudly here is deliberate: the alternative is a silently mis-rendered prompt. */
    static RuntimeException unsupported(String feature) {
        return new RuntimeException("[jinja] unsupported template feature: " + feature);
    }

    // ── Built-in methods on objects ──────────────────────────────

    private static final class Builtins {
        Val items() { return new Val.Func("items", args -> {
            var src = requireObj(args, 0);
            var arr = new ArrayList<Val>();
            for (var e : src.v.entrySet())
                arr.add(new Val.Arr(List.of(new Val.Str(e.getKey()), e.getValue())));
            return new Val.Arr(arr);
        });}
        Val keys() { return new Val.Func("keys", args -> {
            var src = requireObj(args, 0);
            var keys = new ArrayList<Val>();
            for (String k : src.v.keySet()) keys.add(new Val.Str(k));
            return new Val.Arr(keys);
        });}
        Val values() { return new Val.Func("values", args -> {
            var src = requireObj(args, 0);
            return new Val.Arr(new ArrayList<>(src.v.values()));
        });}
        Val get() { return new Val.Func("get", args -> {
            var src = requireObj(args, 0);
            String key = expectStr(requireArg(args, 1));
            Val def = args.size() > 2 ? requireArg(args, 2) : Val.NONE;
            Val found = src.v.get(key);
            return found != null ? found : def;
        });}
    }

    private static final Builtins BUILTIN_METHODS = new Builtins();

    private static Val.Obj requireObj(List<Val> args, int idx) {
        Val v = args.get(idx);
        if (v instanceof Val.Obj o) return o;
        throw new RuntimeException("expected object, got " + v);
    }

    // ── Filters ──────────────────────────────────────────────────

    static Val applyFilter(String name, Val val, List<Val> args) {
        return switch (name) {
            case "tojson"    -> tojson(val);
            case "trim"      -> new Val.Str(val.asStr().strip());
            case "string"    -> new Val.Str(val.asStr());
            case "length"    -> new Val.Int(length(val));
            case "default"   -> val.isDefined() ? val : (args.isEmpty() ? new Val.Str("") : requireArg(args, 0));
            case "join"      -> new Val.Str(join(val, args.isEmpty() ? "" : expectStr(requireArg(args, 0))));
            case "split"     -> split(val, expectStr(requireArg(args, 0)));
            case "upper"     -> new Val.Str(val.asStr().toUpperCase());
            case "lower"     -> new Val.Str(val.asStr().toLowerCase());
            case "startswith" -> new Val.Bool(val.asStr().startsWith(expectStr(requireArg(args, 0))));
            case "endswith"   -> new Val.Bool(val.asStr().endsWith(expectStr(requireArg(args, 0))));
            case "list"      -> toList(val);
            case "first"     -> val instanceof Val.Arr a && !a.v.isEmpty() ? a.v.getFirst() : Val.NONE;
            case "last"      -> val instanceof Val.Arr a && !a.v.isEmpty() ? a.v.getLast() : Val.NONE;
            case "count"     -> new Val.Int(length(val));
            case "capitalize" -> new Val.Str(capitalize(val.asStr()));
            case "replace"   -> new Val.Str(val.asStr().replace(expectStr(requireArg(args, 0)), expectStr(requireArg(args, 1))));
            case "selectattr" -> filterAttr(val, args, true);
            case "rejectattr" -> filterAttr(val, args, false);
            default -> throw unsupported("filter '" + name + "'");
        };
    }

    static long length(Val v) {
        return switch (v) {
            case Val.Str s -> s.v.length();
            case Val.Arr a -> a.v.size();
            case Val.Obj o -> o.v.size();
            default -> v.asStr().length();
        };
    }

    static String join(Val v, String sep) {
        if (v instanceof Val.Arr a) {
            var sb = new StringBuilder();
            for (int i = 0; i < a.v.size(); i++) {
                if (i > 0) sb.append(sep);
                sb.append(a.v.get(i).asStr());
            }
            return sb.toString();
        }
        return v.asStr();
    }

    static Val split(Val v, String delimiter) {
        String s = v.asStr();
        if (s.isEmpty()) return new Val.Arr(List.of());
        var parts = new ArrayList<Val>();
        for (String part : s.split(java.util.regex.Pattern.quote(delimiter), -1))
            parts.add(new Val.Str(part));
        return new Val.Arr(parts);
    }

    /** {@code list} filter: arrays pass through; strings explode into characters; scalars wrap. */
    static Val toList(Val v) {
        if (v instanceof Val.Arr) return v;
        if (v instanceof Val.Str s) {
            var chars = new ArrayList<Val>(s.v.length());
            for (int i = 0; i < s.v.length(); i++) chars.add(new Val.Str(String.valueOf(s.v.charAt(i))));
            return new Val.Arr(chars);
        }
        return new Val.Arr(new ArrayList<>(List.of(v)));
    }

    static String capitalize(String s) {
        return s.isEmpty() ? s : Character.toUpperCase(s.charAt(0)) + s.substring(1).toLowerCase();
    }

    /** {@code selectattr}/{@code rejectattr}: keep (or drop) items whose attribute passes a test.
     *  Forms: {@code seq|selectattr('attr')} (truthy) and {@code seq|selectattr('attr', test, arg)}
     *  (e.g. {@code 'equalto'}). Non-arrays pass through unchanged. */
    static Val filterAttr(Val seq, List<Val> args, boolean keep) {
        if (!(seq instanceof Val.Arr a)) return seq;
        String attr = expectStr(requireArg(args, 0));
        String test = args.size() > 1 ? expectStr(args.get(1)) : null;
        Val cmp = args.size() > 2 ? args.get(2) : null;
        var out = new ArrayList<Val>();
        for (Val item : a.v()) {
            Val av = item instanceof Val.Obj o ? o.get(attr) : Val.NONE;
            boolean match = test == null ? av.truthy() : attrTest(test, av, cmp);
            if (match == keep) out.add(item);
        }
        return new Val.Arr(out);
    }

    static boolean attrTest(String test, Val v, Val cmp) {
        return switch (test) {
            case "equalto", "eq", "==" -> Executor.eq(v, cmp);
            case "ne", "!=" -> !Executor.eq(v, cmp);
            case "in" -> cmp != null && Executor.contains(cmp, v);
            default -> applyTest(test, v); // defined / none / string / number / ...
        };
    }

    static Val tojson(Val v) {
        StringBuilder sb = new StringBuilder();
        writeJson(sb, v);
        return new Val.Str(sb.toString());
    }

    private static void writeJson(StringBuilder sb, Val v) {
        switch (v) {
            case Val.None __ -> sb.append("null");
            case Val.Bool b -> sb.append(b.v);
            case Val.Int i -> sb.append(i.v);
            case Val.Flt f -> sb.append(fmtDouble(f.v));
            case Val.Str s -> sb.append(Json.stringify(s.v));
            case Val.Arr a -> {
                sb.append('[');
                for (int i = 0; i < a.v.size(); i++) {
                    if (i > 0) sb.append(", ");
                    writeJson(sb, a.v.get(i));
                }
                sb.append(']');
            }
            case Val.Obj o -> {
                sb.append('{');
                boolean first = true;
                for (var e : o.v.entrySet()) {
                    if (!first) sb.append(", ");
                    first = false;
                    sb.append(Json.stringify(e.getKey()));
                    sb.append(": ");
                    writeJson(sb, e.getValue());
                }
                sb.append('}');
            }
            case Val.Func f -> sb.append('"').append(f.name).append('"');
            case Val.Undef u -> sb.append("null");
            default -> sb.append("null");
        }
    }

    // ── Functions ────────────────────────────────────────────────

    static Val callFunction(String name, List<Val> args) {
        return switch (name) {
            // note: namespace(...) is handled in CallNode evaluation so its kwargs keep their
            // value types (this string-args path would coerce everything to strings).
            case "format_time", "strftime_now" -> {
                String fmt = args.isEmpty() ? "%Y-%m-%d" : expectStr(requireArg(args, 0));
                yield new Val.Str(strftime(java.time.LocalDateTime.now(), fmt));
            }
            // chat templates call raise_exception(...) to reject malformed conversations; surface
            // it as a distinct error (not an "unsupported feature") so the message is the real one.
            case "raise_exception" -> throw new RuntimeException("[jinja:raise] "
                    + (args.isEmpty() ? "template raised an exception" : expectStr(requireArg(args, 0))));
            default -> throw unsupported("function '" + name + "()'");
        };
    }

    /** Minimal C/Python strftime: handles the directives chat templates actually use. */
    static String strftime(java.time.LocalDateTime t, String fmt) {
        var sb = new StringBuilder();
        for (int i = 0; i < fmt.length(); i++) {
            char c = fmt.charAt(i);
            if (c != '%' || i + 1 >= fmt.length()) { sb.append(c); continue; }
            char d = fmt.charAt(++i);
            sb.append(switch (d) {
                case 'Y' -> String.format("%04d", t.getYear());
                case 'y' -> String.format("%02d", t.getYear() % 100);
                case 'm' -> String.format("%02d", t.getMonthValue());
                case 'd' -> String.format("%02d", t.getDayOfMonth());
                case 'H' -> String.format("%02d", t.getHour());
                case 'I' -> String.format("%02d", (t.getHour() % 12 == 0) ? 12 : t.getHour() % 12);
                case 'M' -> String.format("%02d", t.getMinute());
                case 'S' -> String.format("%02d", t.getSecond());
                case 'p' -> t.getHour() < 12 ? "AM" : "PM";
                case 'j' -> String.format("%03d", t.getDayOfYear());
                case 'B' -> t.getMonth().getDisplayName(java.time.format.TextStyle.FULL, java.util.Locale.ENGLISH);
                case 'b' -> t.getMonth().getDisplayName(java.time.format.TextStyle.SHORT, java.util.Locale.ENGLISH);
                case 'A' -> t.getDayOfWeek().getDisplayName(java.time.format.TextStyle.FULL, java.util.Locale.ENGLISH);
                case 'a' -> t.getDayOfWeek().getDisplayName(java.time.format.TextStyle.SHORT, java.util.Locale.ENGLISH);
                case '%' -> "%";
                default -> "%" + d;
            });
        }
        return sb.toString();
    }

    // ── Tests ────────────────────────────────────────────────────

    static boolean applyTest(String test, Val val) {
        return switch (test) {
            case "defined"  -> val.isDefined();
            case "undefined" -> !val.isDefined();
            case "none"     -> val.isNone();
            case "string"   -> val instanceof Val.Str;
            case "mapping"  -> val instanceof Val.Obj;
            case "sequence" -> val instanceof Val.Arr;
            case "number"   -> val instanceof Val.Int || val instanceof Val.Flt;
            case "boolean"  -> val instanceof Val.Bool;
            case "callable" -> val instanceof Val.Func;
            case "iterable" -> val instanceof Val.Arr || val instanceof Val.Obj;
            case "true"     -> val.truthy();
            case "false"    -> !val.truthy();
            case "odd"      -> (val instanceof Val.Int i && i.v % 2 != 0);
            case "even"     -> (val instanceof Val.Int i && i.v % 2 == 0);
            case "integer"  -> val instanceof Val.Int;
            case "float"    -> val instanceof Val.Flt;
            default -> false;
        };
    }

    private static Val requireArg(List<Val> args, int idx) {
        if (idx < args.size()) return args.get(idx);
        throw new RuntimeException("missing argument at index " + idx);
    }

    private static String expectStr(Val v) {
        if (v instanceof Val.Str s) return s.v;
        return v.asStr();
    }

    // ════════════════════════════════════════════════════════════════
    // TOKEN TYPES
    // ════════════════════════════════════════════════════════════════

    enum T { EOF, TEXT, OPEN_STMT, CLOSE_STMT, OPEN_EXPR, CLOSE_EXPR,
             IDENT, STRING, INT, FLOAT,
             // Punctuation
             EQ, LP, RP, LB, RB, LC, RC, COMMA, DOT, COLON, PIPE, TILDE,
             // Operators
             PLUS, MINUS, STAR, SLASH, PCT,
             // Comparisons (single token: symbol + value string)
             CMP,
             // Unary
             NOT,
    }

    record Tok(T type, String val, int pos, boolean trimLeft, boolean trimRight) {
        Tok(T type, String val, int pos) { this(type, val, pos, false, false); }
    }

    // ════════════════════════════════════════════════════════════════
    // LEXER
    // ════════════════════════════════════════════════════════════════

    private static final class Lexer {
        final CharSequence cs;
        int i;
        int depth; // 0=text mode, >0 inside {{ or {%
        boolean stripNextText; // set when previous close tag had trimRight
        List<Tok> toks; // set by tokenize()

        Lexer(CharSequence cs) { this.cs = cs; }

        char ch()        { return i < cs.length() ? cs.charAt(i) : 0; }
        char ch(int off) { int p = i + off; return p < cs.length() ? cs.charAt(p) : 0; }
        void adv()       { i++; }
        void adv(int n)  { i += n; }
        int pos()        { return i; }
        boolean eof()    { return i >= cs.length(); }

        List<Tok> tokenize() {
            toks = new ArrayList<Tok>();
            while (!eof()) {
                int p = pos();
                char c = ch();

                // Text mode: accumulate everything until {{
                if (depth == 0) {
                    if (c == '{' && (ch(1) == '%' || ch(1) == '{' || ch(1) == '#')) {
                        // Tag opener
                        if (matchTrim(T.OPEN_STMT, "{%-")) { depth++; }
                        else if (match(T.OPEN_STMT, "{%")) { depth++; toks.add(tok(T.OPEN_STMT, p)); }
                        else if (matchTrim(T.OPEN_EXPR, "{{-")) { depth++; }
                        else if (match(T.OPEN_EXPR, "{{")) { depth++; toks.add(tok(T.OPEN_EXPR, p)); }
                        else if (matchStr("{#")) {
                            while (!eof() && !(ch() == '#' && ch(1) == '}')) adv();
                            if (!eof()) adv(2);
                        } else { adv(); } // stray {
                    } else {
                        toks.add(readText(p));
                    }
                    continue;
                }

                // Inside {{ or {% — skip whitespace, tokenize individually
                if (Character.isWhitespace(c)) { adv(); continue; }

                // Close tags (must check before other token types)
                if (matchTrim(T.CLOSE_STMT, "-%}")) { depth--; continue; }
                if (match(T.CLOSE_STMT, "%}")) { depth--; toks.add(tok(T.CLOSE_STMT, p)); continue; }
                if (matchTrim(T.CLOSE_EXPR, "-}}")) { depth--; continue; }
                if (match(T.CLOSE_EXPR, "}}")) { depth--; toks.add(tok(T.CLOSE_EXPR, p)); continue; }

                // Two-char comparisons
                if (matchStr("<=") || matchStr(">=") || matchStr("==") || matchStr("!=")) { toks.add(new Tok(T.CMP, cs.subSequence(p, i).toString(), p)); continue; }
                if (c == '<' || c == '>') { adv(); toks.add(new Tok(T.CMP, String.valueOf(c), p)); continue; }

                // Punctuation/operators
                switch (c) {
                    case '=' -> { adv(); toks.add(new Tok(T.EQ, "=", p)); }
                    case '(' -> { adv(); toks.add(new Tok(T.LP, "(", p)); }
                    case ')' -> { adv(); toks.add(new Tok(T.RP, ")", p)); }
                    case '[' -> { adv(); toks.add(new Tok(T.LB, "[", p)); }
                    case ']' -> { adv(); toks.add(new Tok(T.RB, "]", p)); }
                    case '{' -> { adv(); toks.add(new Tok(T.LC, "{", p)); }
                    case '}' -> { adv(); toks.add(new Tok(T.RC, "}", p)); }
                    case ',' -> { adv(); toks.add(new Tok(T.COMMA, ",", p)); }
                    case '.' -> { adv(); toks.add(new Tok(T.DOT, ".", p)); }
                    case ':' -> { adv(); toks.add(new Tok(T.COLON, ":", p)); }
                    case '|' -> { adv(); toks.add(new Tok(T.PIPE, "|", p)); }
                    case '~' -> { adv(); toks.add(new Tok(T.TILDE, "~", p)); }
                    case '+' -> { adv(); toks.add(new Tok(T.PLUS, "+", p)); }
                    case '-' -> { adv(); toks.add(new Tok(T.MINUS, "-", p)); }
                    case '*' -> { adv(); toks.add(new Tok(T.STAR, "*", p)); }
                    case '/' -> { adv(); toks.add(new Tok(T.SLASH, "/", p)); }
                    case '%' -> { adv(); toks.add(new Tok(T.PCT, "%", p)); }
                    default -> {}
                }
                if (i > p) continue; // token emitted

                // Number
                if (Character.isDigit(c) || (c == '.' && Character.isDigit(ch(1)))) {
                    toks.add(readNumber(p));
                    continue;
                }

                // String literal
                if (c == '"' || c == '\'') { toks.add(readString(p, c)); continue; }

                // Identifier
                if (Character.isLetter(c) || c == '_') { toks.add(readIdent(p)); continue; }

                // Stray char — treat as text
                toks.add(new Tok(T.TEXT, String.valueOf(c), p));
                adv();
            }
            toks.add(new Tok(T.EOF, "", i));
            return toks;
        }

        boolean match(T type, String s) {
            for (int j = 0; j < s.length(); j++)
                if (ch(j) != s.charAt(j)) return false;
            adv(s.length());
            return true;
        }

        boolean matchTrim(T type, String s) {
            // s is e.g. "{%-" or "-}}" — the FULL marker including trim dashes
            for (int j = 0; j < s.length(); j++)
                if (ch(j) != s.charAt(j)) return false;
            int save = pos();
            adv(s.length());
            boolean left = s.startsWith("-"), right = s.endsWith("-");
            if (left) stripLastText();
            if (right) stripNextText = true;
            toks.add(new Tok(type, "", save, left, right));
            return true;
        }

        /** Strip trailing whitespace from the last emitted TEXT token. */
        void stripLastText() {
            for (int j = toks.size() - 1; j >= 0; j--) {
                Tok pt = toks.get(j);
                if (pt.type == T.TEXT) {
                    String stripped = pt.val.stripTrailing();
                    if (!stripped.equals(pt.val))
                        toks.set(j, new Tok(T.TEXT, stripped, pt.pos));
                    break;
                }
            }
        }

        boolean matchStr(String s) {
            for (int j = 0; j < s.length(); j++)
                if (ch(j) != s.charAt(j)) return false;
            adv(s.length());
            return true;
        }

        Tok tok(T type, int p) { return new Tok(type, "", p); }

        Tok readNumber(int p) {
            if (ch() == '-') adv();
            while (Character.isDigit(ch())) adv();
            boolean isFloat = ch() == '.' && Character.isDigit(ch(1));
            if (isFloat) { adv(); while (Character.isDigit(ch())) adv(); }
            String n = cs.subSequence(p, i).toString();
            return isFloat ? new Tok(T.FLOAT, n, p) : new Tok(T.INT, n, p);
        }

        Tok readString(int p, char q) {
            adv(); // opening quote
            var sb = new StringBuilder();
            while (!eof() && ch() != q) {
                if (ch() == '\\' && !eof()) {
                    adv(); // skip backslash
                    if (!eof()) {
                        sb.append(switch (ch()) {
                            case 'n' -> '\n'; case 't' -> '\t'; case 'r' -> '\r';
                            case '\\' -> '\\'; case '\'' -> '\''; case '"' -> '"';
                            default -> { sb.append('\\'); yield ch(); }
                        });
                    }
                } else {
                    sb.append(ch());
                }
                adv();
            }
            if (!eof()) adv(); // closing quote
            return new Tok(T.STRING, sb.toString(), p);
        }

        Tok readIdent(int p) {
            int s = i;
            while (!eof() && (Character.isLetterOrDigit(ch()) || ch() == '_')) adv();
            return new Tok(T.IDENT, cs.subSequence(s, i).toString(), p);
        }

        Tok readText(int p) {
            int s = i;
            while (!eof()) {
                char c = ch();
                if (c == '{') {
                    char n = ch(1);
                    if (n == '%' || n == '{' || n == '#') break;
                }
                adv();
            }
            if (i == s) { // shouldn't happen — but guard against it
                if (eof()) return new Tok(T.EOF, "", s);
                adv(); // skip the { that triggered the stop
            }
            String val = cs.subSequence(s, i).toString();
            if (stripNextText) { val = val.stripLeading(); stripNextText = false; }
            return new Tok(T.TEXT, val, p);
        }
    }

    // ════════════════════════════════════════════════════════════════
    // AST NODES
    // ════════════════════════════════════════════════════════════════

    public sealed interface Node {}
    public record Prog(List<Node> body) implements Node {}
    record TextNode(String s) implements Node {}
    record OutputNode(Node expr) implements Node {}
    record IfNode(Node test, List<Node> body, List<Node> alt) implements Node {}
    record CondNode(Node test, Node then, Node orElse) implements Node {} // a if test else b (value-valued)
    record ForNode(String var, String var2, Node iterable, List<Node> body) implements Node {}
    record SetNode(Node target, Node value) implements Node {}
    record MacroNode(String name, List<String> params, List<Node> defaults, List<Node> body) implements Node {}
    record FilterNode(Node operand, String name, List<Node> args) implements Node {}
    record BinNode(String op, Node left, Node right) implements Node {}
    record UnaNode(String op, Node arg) implements Node {}
    record MemberNode(Node obj, Node prop, boolean computed) implements Node {}
    record CallNode(Node callee, List<Node> args) implements Node {}
    record IdentNode(String name) implements Node {}
    record LitNode(Object val) implements Node {} // String, Long, Double, Boolean, Val.None
    record TestNode(String name, Node operand, boolean negate) implements Node {}
    record SliceNode(Node start, Node stop, Node step) implements Node {}

    // ════════════════════════════════════════════════════════════════
    // PARSER (token-based, like llama.cpp)
    // ════════════════════════════════════════════════════════════════

    /** Compiles the template source into an executable AST. */
    public static Prog compile(String source) {
        return new Parser(new Lexer(source).tokenize()).parseProgram();
    }

    private static final class Parser {
        final List<Tok> toks;
        int idx;

        Parser(List<Tok> toks) { this.toks = toks; }

        Tok peek() { return idx < toks.size() ? toks.get(idx) : eofTok(); }
        Tok peek(int off) { int p = idx + off; return p < toks.size() ? toks.get(p) : eofTok(); }
        Tok next()  { return toks.get(idx++); }
        boolean eof() { return idx >= toks.size() || peek().type == T.EOF; }
        static Tok eofTok() { return new Tok(T.EOF, "", 0); }

        Tok expect(T type) {
            Tok t = peek();
            if (t.type != type) throw err("expected " + type + ", got " + t.type + " '" + t.val + "'");
            idx++;
            return t;
        }

        void expectId(String name) {
            Tok t = peek();
            if (t.type != T.IDENT || !t.val.equals(name))
                throw err("expected identifier '" + name + "', got " + t);
            idx++;
        }

        boolean is(T type) { return peek().type == type; }
        boolean isId(String name) { return peek().type == T.IDENT && peek().val.equals(name); }

        boolean isStmt(String... names) {
            return peek().type == T.OPEN_STMT && peek(1).type == T.IDENT
                   && List.of(names).contains(peek(1).val);
        }

        RuntimeException err(String msg) {
            Tok t = peek();
            return new RuntimeException("@" + t.pos + ": " + msg + " near: '" + t.val + "'");
        }

        Prog parseProgram() {
            var body = new ArrayList<Node>();
            while (!eof()) body.add(parseAny());
            return new Prog(body);
        }

        Node parseAny() {
            Tok t = peek();
            return switch (t.type) {
                case T.TEXT -> { idx++; yield new TextNode(t.val); }
                case T.OPEN_EXPR -> { idx++; Node e = parseExpr(); expect(T.CLOSE_EXPR); yield new OutputNode(e); }
                case T.OPEN_STMT -> parseStatement();
                default           -> { idx++; yield new TextNode(t.val); } // comment or stray
            };
        }

        Node parseStatement() {
            expect(T.OPEN_STMT);
            String kw = peek().type == T.IDENT ? next().val : "";
            Node result;
            switch (kw) {
                case "if"    -> result = parseIf();
                case "for"   -> result = parseFor();
                case "set"   -> result = parseSet();
                case "macro" -> result = parseMacro();
                case "generation", "endgeneration" -> {
                    // transformers-specific extension — transparent no-op
                    result = new TextNode("");
                    expect(T.CLOSE_STMT);
                    return result;
                }
                case "break", "continue" -> throw err("{% " + kw + " %} is not supported");
                case "else", "elif", "endif", "endfor", "endmacro", "end" -> {
                    // These are handled by enclosing parse functions
                    idx--; // back up over the identifier
                    return new TextNode("");
                }
                default -> {
                    // Expression statement: {% expr %}
                    // Back up the identifier so parseExpr picks it up
                    idx--;
                    Node expr = parseExpr();
                    expect(T.CLOSE_STMT);
                    result = new OutputNode(expr);
                }
            }
            return result;
        }

        // ── if ──

        Node parseIf() {
            Node test = parseExpr();
            expect(T.CLOSE_STMT);
            List<Node> body = parseBody("elif", "else", "endif");
            return new IfNode(test, body, parseIfTail());
        }

        /** The chain after an {@code if}/{@code elif} body: each {@code elif} nests as the single
         *  alternative of the previous branch (so they stay mutually exclusive and a trailing
         *  {@code else} is preserved rather than clobbering the elifs), terminated by {@code endif}. */
        List<Node> parseIfTail() {
            if (isStmt("elif")) {
                expect(T.OPEN_STMT); expectId("elif");
                Node test = parseExpr();
                expect(T.CLOSE_STMT);
                List<Node> body = parseBody("elif", "else", "endif");
                return new ArrayList<>(List.of(new IfNode(test, body, parseIfTail())));
            }
            List<Node> alt = new ArrayList<>();
            if (isStmt("else")) {
                expect(T.OPEN_STMT); expectId("else"); expect(T.CLOSE_STMT);
                alt = parseBody("endif");
            }
            expect(T.OPEN_STMT); expectId("endif"); expect(T.CLOSE_STMT);
            return alt;
        }

        // ── for ──

        Node parseFor() {
            var loopVars = parseVarList();
            expectId("in");
            Node iterable = parseExpr();
            expect(T.CLOSE_STMT);
            var body = parseBody("endfor");
            expect(T.OPEN_STMT); expectId("endfor"); expect(T.CLOSE_STMT);
            String v1 = loopVars.getFirst();
            String v2 = loopVars.size() > 1 ? loopVars.get(1) : null;
            return new ForNode(v1, v2, iterable, body);
        }

        List<String> parseVarList() {
            var vars = new ArrayList<String>();
            vars.add(expect(T.IDENT).val);
            while (is(T.COMMA)) { next(); vars.add(expect(T.IDENT).val); }
            return vars;
        }

        // ── set ──

        Node parseSet() {
            Node target = parseExpr();
            Node value = null;
            if (is(T.EQ)) { next(); value = parseExpr(); }
            expect(T.CLOSE_STMT);
            return new SetNode(target, value);
        }

        // ── macro ──

        Node parseMacro() {
            String name = expect(T.IDENT).val;
            expect(T.LP);
            var params = new ArrayList<String>();
            var defaults = new ArrayList<Node>(); // parallel to params; null = no default
            while (!is(T.RP)) {
                params.add(expect(T.IDENT).val);
                defaults.add(is(T.EQ) ? consumeDefault() : null);
                if (is(T.COMMA)) next();
            }
            expect(T.RP);
            expect(T.CLOSE_STMT);
            var body = parseBody("endmacro");
            expect(T.OPEN_STMT); expectId("endmacro"); expect(T.CLOSE_STMT);
            return new MacroNode(name, params, defaults, body);
        }

        private Node consumeDefault() {
            next(); // '='
            return parseExpr();
        }

        // ── body parsing ──

        List<Node> parseBody(String... stopKeywords) {
            var body = new ArrayList<Node>();
            while (!eof() && !isStmt(stopKeywords)) {
                body.add(parseAny());
            }
            return body;
        }

        // ── expressions ──

        Node parseExpr() {
            skipText();
            return parseTernary();
        }

        void skipText() { while (!eof() && is(T.TEXT)) idx++; }

        Node parseTernary() {
            Node thenVal = parseOr();
            if (isId("if")) {
                next(); // if
                Node test = parseOr();
                Node orElse = null; // omitted else -> undefined when the test is false
                if (isId("else")) { next(); orElse = parseTernary(); }
                return new CondNode(test, thenVal, orElse);
            }
            return thenVal;
        }

        Node parseOr() {
            Node left = parseAnd();
            while (isId("or")) { next(); left = new BinNode("or", left, parseAnd()); }
            return left;
        }

        Node parseAnd() {
            Node left = parseNot();
            while (isId("and")) { next(); left = new BinNode("and", left, parseNot()); }
            return left;
        }

        Node parseNot() {
            if (isId("not")) { next(); return new UnaNode("not", parseNot()); }
            return parseComp();
        }

        Node parseComp() {
            Node left = parseConcat();
            // is / is not
            if (isId("is")) {
                next(); boolean neg = false;
                if (isId("not")) { next(); neg = true; }
                String test = expect(T.IDENT).val;
                return new TestNode(test, left, neg);
            }
            // in / not in
            if (isId("in")) { next(); return new BinNode("in", left, parseConcat()); }
            if (isId("not") && peek(1).type == T.IDENT && peek(1).val.equals("in")) {
                next(); next(); return new BinNode("notin", left, parseConcat());
            }
            // comparison operators
            if (is(T.CMP)) { String op = next().val; return new BinNode(op, left, parseConcat()); }
            return left;
        }

        /** String concatenation with {@code ~}, left-associative and chainable (a ~ b ~ c),
         *  binding tighter than comparisons but looser than +/-. */
        Node parseConcat() {
            Node left = parseAdd();
            while (is(T.TILDE)) { next(); left = new BinNode("~", left, parseAdd()); }
            return left;
        }

        Node parseAdd() {
            Node left = parseMul();
            while (true) {
                if (is(T.PLUS)) { next(); left = new BinNode("+", left, parseMul()); continue; }
                if (is(T.MINUS)) { next(); left = new BinNode("-", left, parseMul()); continue; }
                break;
            }
            return left;
        }

        Node parseMul() {
            Node left = parseUnary();
            while (true) {
                if (is(T.STAR)) { next(); left = new BinNode("*", left, parseUnary()); continue; }
                if (is(T.SLASH)) { next(); left = new BinNode("/", left, parseUnary()); continue; }
                if (is(T.PCT)) { next(); left = new BinNode("%", left, parseUnary()); continue; }
                break;
            }
            return left;
        }

        Node parseUnary() {
            if (is(T.NOT)) { next(); return new UnaNode("not", parseUnary()); }
            if (is(T.PLUS)) { next(); return parseUnary(); } // unary plus is no-op
            if (is(T.MINUS)) { next(); return new BinNode("*", new LitNode(-1L), parseUnary()); }
            return parseTestExpr();
        }

        Node parseTestExpr() {
            Node operand = parseFilter();
            if (isId("is")) {
                next(); boolean neg = false;
                if (isId("not")) { next(); neg = true; }
                String test = expect(T.IDENT).val;
                return new TestNode(test, operand, neg);
            }
            return operand;
        }

        Node parseFilter() {
            Node left = parseCall();
            while (is(T.PIPE)) {
                next(); // |
                String name = expect(T.IDENT).val;
                List<Node> args = new ArrayList<>();
                if (is(T.LP)) {
                    next();
                    if (!is(T.RP)) args = new ArrayList<>(parseArgList());
                    expect(T.RP);
                }
                left = new FilterNode(left, name, args);
            }
            return left;
        }

        Node parseCall() {
            Node node = parseAtom();
            while (true) {
                if (is(T.LP)) {
                    next();
                    var args = is(T.RP) ? new ArrayList<Node>() : new ArrayList<>(parseArgList());
                    expect(T.RP);
                    node = new CallNode(node, args);
                    continue;
                }
                if (is(T.DOT)) {
                    next();
                    String prop = expect(T.IDENT).val;
                    node = new MemberNode(node, new LitNode(prop), false);
                    continue;
                }
                if (is(T.LB)) {
                    next();
                    Node key;
                    if (is(T.COLON)) {
                        key = new LitNode(Val.NONE); // empty start for [:end]
                    } else {
                        key = parseExpr();
                    }
                    if (is(T.COLON)) {
                        next();
                        // an immediate ']' , ',' or ':' means the stop bound was omitted ([n:], [::s])
                        Node stop = (is(T.RB) || is(T.COMMA) || is(T.COLON)) ? new LitNode(Val.NONE) : parseExpr();
                        Node step = new LitNode(Val.NONE);
                        if (is(T.COLON)) { next(); step = is(T.RB) ? new LitNode(Val.NONE) : parseExpr(); }
                        expect(T.RB);
                        node = new MemberNode(node, new SliceNode(key, stop, step), true);
                    } else {
                        expect(T.RB);
                        node = new MemberNode(node, key, true);
                    }
                    continue;
                }
                break;
            }
            return node;
        }

        Node parseAtom() {
            Tok t = next();
            return switch (t.type) {
                case T.IDENT -> {
                    Node id = new IdentNode(t.val);
                    if (is(T.LP)) {
                        next();
                        var args = is(T.RP) ? new ArrayList<Node>() : new ArrayList<>(parseArgList());
                        expect(T.RP);
                        yield new CallNode(id, args);
                    }
                    yield id;
                }
                case T.STRING -> {
                    // adjacent string literals concatenate, like Python/Jinja2 ('a' 'b' -> 'ab')
                    StringBuilder sb = new StringBuilder(t.val);
                    while (is(T.STRING)) sb.append(next().val);
                    yield new LitNode(sb.toString());
                }
                case T.INT -> {
                    try { yield new LitNode(Long.parseLong(t.val)); }
                    catch (NumberFormatException e) { yield new LitNode(0L); }
                }
                case T.FLOAT -> {
                    try { yield new LitNode(Double.parseDouble(t.val)); }
                    catch (NumberFormatException e) { yield new LitNode(0.0); }
                }
                case T.LP -> {
                    Node e = parseExpr();
                    if (is(T.COMMA)) { // tuple literal -> treated as a list
                        var items = new ArrayList<Node>();
                        items.add(e);
                        while (is(T.COMMA)) { next(); if (is(T.RP)) break; items.add(parseExpr()); }
                        expect(T.RP);
                        yield new LitNode(items);
                    }
                    expect(T.RP);
                    yield e;
                }
                case T.LB -> {
                    if (is(T.RB)) { next(); yield new LitNode(new ArrayList<Node>()); }
                    var items = new ArrayList<Node>();
                    items.add(parseExpr());
                    while (is(T.COMMA)) { next(); if (is(T.RB)) break; items.add(parseExpr()); }
                    expect(T.RB);
                    yield new LitNode(items);
                }
                case T.LC -> { yield parseDictExpr(); }
                default -> throw err("unexpected token " + t.type);
            };
        }

        Node parseDictExpr() {
            // { already consumed
            if (is(T.RC)) { next(); return new LitNode(new LinkedHashMap<String,Node>()); }
            var map = new LinkedHashMap<String,Node>();
            while (true) {
                Node key = parseExpr();
                expect(T.COLON);
                Node val = parseExpr();
                // string-literal and identifier keys are the common forms; fall back to the
                // literal's value for number keys
                String k = key instanceof IdentNode i ? i.name()
                         : key instanceof LitNode lit ? String.valueOf(lit.val())
                         : key.toString();
                map.put(k, val);
                if (is(T.COMMA)) { next(); if (is(T.RC)) break; }
                else break;
            }
            expect(T.RC);
            return new LitNode(map);
        }

        List<Node> parseArgList() {
            var args = new ArrayList<Node>();
            while (true) {
                args.add(parseExpr());
                // Keyword argument: name=value
                if (is(T.EQ)) {
                    // Replace the last arg (the name) with a BinNode wrapper
                    Node name = args.removeLast();
                    next(); // consume =
                    args.add(new BinNode("kwarg", name, parseExpr()));
                }
                if (is(T.COMMA)) { next(); continue; }
                break;
            }
            return args;
        }
    }

    // ════════════════════════════════════════════════════════════════
    // TEMPLATE EXECUTOR (unchanged from previous version)
    // ════════════════════════════════════════════════════════════════

    public static String render(Prog program, Map<String,Object> context) {
        var ctx = new Frame();
        for (var e : context.entrySet()) ctx.set(e.getKey(), Val.of(e.getValue()));
        return new Executor(ctx, new ArrayList<>()).execute(program);
    }

    public static String render(String template, Map<String,Object> context) {
        return render(compile(template), context);
    }

    static final class Frame {
        final LinkedHashMap<String,Val> vars = new LinkedHashMap<>();
        // macro definitions live on the (shared) root frame so calls can bind params by name,
        // honoring defaults and keyword arguments — see Executor.CallNode handling
        final Map<String,MacroNode> macros = new java.util.HashMap<>();
        Val get(String name) {
            Val v = vars.get(name);
            if (v != null) return v;
            if (name.equals("loop")) return vars.get("__loop__");
            // Jinja2 accepts both capitalized and lowercase literals
            if (name.equals("True") || name.equals("true")) return new Val.Bool(true);
            if (name.equals("False") || name.equals("false")) return new Val.Bool(false);
            if (name.equals("None") || name.equals("none") || name.equals("null")) return Val.NONE;
            return new Val.Undef(name);
        }
        void set(String name, Val val) { vars.put(name, val); }
    }

    static final class Executor {
        final Frame frame;
        final List<Frame> stack;

        Executor(Frame root, List<Frame> stack) { this.frame = root; this.stack = stack; }

        String execute(Prog prog) {
            var out = new StringBuilder();
            for (Node n : prog.body()) exec(n, out);
            return out.toString();
        }

        void exec(Node node, StringBuilder out) {
            switch (node) {
                case TextNode t -> out.append(t.s());
                case OutputNode o -> out.append(eval(o.expr()).asStr());
                case IfNode i -> {
                    if (eval(i.test()).truthy()) execAll(i.body(), out);
                    else execAll(i.alt(), out);
                }
                case ForNode f -> {
                    Val iter = eval(f.iterable());
                    List<Val> items = switch (iter) {
                        case Val.Arr a -> a.v();
                        case Val.Obj o -> {
                            var arr = new ArrayList<Val>();
                            for (var e : o.v.entrySet())
                                arr.add(new Val.Arr(List.of(new Val.Str(e.getKey()), e.getValue())));
                            yield arr;
                        }
                        default -> {
                            String s = iter.asStr();
                            if (s.isEmpty()) yield List.of();
                            var arr = new ArrayList<Val>();
                            for (String part : s.split("\n")) arr.add(new Val.Str(part));
                            yield arr;
                        }
                    };
                    for (int idx = 0; idx < items.size(); idx++) {
                        Val item = items.get(idx);
                        var lf = new Frame();
                        if (f.var2() != null && item instanceof Val.Arr a && a.v.size() >= 2) {
                            lf.set(f.var(), a.v.get(0));
                            lf.set(f.var2(), a.v.get(1));
                        } else {
                            lf.set(f.var(), item);
                        }
                        var loop = new LinkedHashMap<String,Val>();
                        loop.put("index0", new Val.Int(idx));
                        loop.put("index",  new Val.Int(idx + 1));
                        loop.put("first",  new Val.Bool(idx == 0));
                        loop.put("last",   new Val.Bool(idx == items.size() - 1));
                        loop.put("length", new Val.Int(items.size()));
                        lf.set("__loop__", new Val.Obj(loop, false));
                        try {
                            stack.addLast(lf);
                            execAll(f.body(), out);
                        } catch (BreakSignal __) { break; }
                        catch (ContinueSignal __) { continue; }
                        finally { stack.removeLast(); }
                    }
                }
                case SetNode s -> {
                    Val v = eval(s.value());
                    if (s.target() instanceof IdentNode id)
                        scope().set(id.name(), v);
                    else if (s.target() instanceof MemberNode mm) {
                        Val obj = eval(mm.obj());
                        if (obj instanceof Val.Obj o && mm.prop() instanceof LitNode lp) {
                            o.set(String.valueOf(lp.val()), v);
                        }
                    }
                }
                case MacroNode m -> {
                    // register for name-based calls (with kwargs/defaults) ...
                    frame.macros.put(m.name(), m);
                    // ... and expose a positional Func value so `macro is defined` / passing it
                    // around still works
                    scope().set(m.name(), new Val.Func(m.name(), args -> invokeMacro(m, args, Map.of())));
                }
                default -> {}
            }
        }

        private static final class BreakSignal extends RuntimeException {}
        private static final class ContinueSignal extends RuntimeException {}

        /** Invoke a macro, binding each parameter from (in priority order) a positional argument,
         *  a keyword argument, its default expression, or undefined. */
        Val invokeMacro(MacroNode m, List<Val> args, Map<String,Val> kwargs) {
            var mf = new Frame();
            for (int i = 0; i < m.params().size(); i++) {
                String p = m.params().get(i);
                Val v;
                if (i < args.size()) v = args.get(i);
                else if (kwargs.containsKey(p)) v = kwargs.get(p);
                else if (m.defaults().get(i) != null) v = eval(m.defaults().get(i));
                else v = Val.NONE;
                mf.set(p, v);
            }
            var ms = new ArrayList<Frame>();
            ms.add(mf);
            return new Executor(frame, ms).evalExprBlock(m.body());
        }

        void execAll(List<Node> nodes, StringBuilder out) {
            for (Node n : nodes) exec(n, out);
        }

        Val evalExprBlock(List<Node> nodes) {
            var out = new StringBuilder();
            execAll(nodes, out);
            return new Val.Str(out.toString());
        }

        Frame scope() { return stack.isEmpty() ? frame : stack.getLast(); }

        Val eval(Node expr) {
            return switch (expr) {
                case LitNode l   -> {
                    // list/dict/tuple literals carry element Nodes — evaluate them here
                    if (l.val() instanceof List<?> li) {
                        var out = new ArrayList<Val>(li.size());
                        for (Object e : li) out.add(e instanceof Node n ? eval(n) : Val.of(e));
                        yield new Val.Arr(out);
                    }
                    if (l.val() instanceof Map<?,?> mp) {
                        var out = new LinkedHashMap<String,Val>();
                        for (var e : mp.entrySet())
                            out.put(String.valueOf(e.getKey()), e.getValue() instanceof Node n ? eval(n) : Val.of(e.getValue()));
                        yield new Val.Obj(out, false);
                    }
                    yield Val.of(l.val());
                }
                case IdentNode id -> lookup(id.name());
                case BinNode b   -> evalBin(b.op(), eval(b.left()), eval(b.right()));
                case UnaNode u   -> evalUnary(u.op(), eval(u.arg()));
                case FilterNode f -> {
                    Val v = eval(f.operand());
                    var a = new ArrayList<Val>();
                    for (Node n : f.args()) a.add(eval(n));
                    yield applyFilter(f.name(), v, a);
                }
                case MemberNode m -> evalMember(eval(m.obj()), m);
                case CallNode c -> {
                    Val callee = eval(c.callee());
                    var args = new ArrayList<Val>();
                    var kwargs = new LinkedHashMap<String,Val>();
                    for (Node n : c.args()) {
                        if (n instanceof BinNode b && "kwarg".equals(b.op())) {
                            String k = b.left() instanceof IdentNode id ? id.name() : eval(b.left()).asStr();
                            kwargs.put(k, eval(b.right()));
                        } else {
                            args.add(eval(n));
                        }
                    }
                    String calleeName = c.callee() instanceof IdentNode id ? id.name() : null;
                    // user macros bind by name so defaults and keyword arguments work
                    if (calleeName != null && frame.macros.containsKey(calleeName))
                        yield invokeMacro(frame.macros.get(calleeName), args, kwargs);
                    // namespace(...) keeps its kwargs as a typed object, so numeric/boolean fields
                    // survive (e.g. `{% set ns = namespace(count=0) %}` stays an int, letting
                    // `ns.count + 1` add rather than concatenate "0" + "1").
                    if (!(callee instanceof Val.Func) && "namespace".equals(calleeName))
                        yield new Val.Obj(new LinkedHashMap<>(kwargs), false);
                    // Other callables take kwargs as trailing "name=value" strings (legacy shape).
                    for (var e : kwargs.entrySet())
                        args.add(new Val.Str(e.getKey() + "=" + e.getValue().asStr()));
                    if (callee instanceof Val.Func f) {
                        yield f.fn().call(args);
                    }
                    if (calleeName != null) {
                        yield callFunction(calleeName, args);
                    }
                    // Calling the result of a member access that isn't a function — e.g. a string
                    // method written WITH parentheses (`s.strip()`); the paren-less `s.strip` works.
                    throw unsupported("call of a non-function value (string/array methods must be used without parentheses, e.g. `s.strip` not `s.strip()`)");
                }
                case TestNode t -> {
                    Val v = eval(t.operand());
                    boolean r = applyTest(t.name(), v);
                    yield new Val.Bool(t.negate() ? !r : r);
                }
                case OutputNode o -> eval(o.expr());
                // ternary: yields the chosen branch's VALUE (not its rendered text); a missing
                // else yields undefined, per Jinja
                case CondNode c -> eval(c.test()).truthy() ? eval(c.then())
                    : (c.orElse() == null ? new Val.Undef("") : eval(c.orElse()));
                case IfNode i -> eval(i.test()).truthy()
                    ? evalBlockExpr(i.body()) : evalBlockExpr(i.alt());
                default -> Val.NONE;
            };
        }

        Val evalBlockExpr(List<Node> nodes) {
            var out = new StringBuilder();
            execAll(nodes, out);
            return new Val.Str(out.toString());
        }

        Val lookup(String name) {
            for (int i = stack.size() - 1; i >= 0; i--) {
                Val v = stack.get(i).get(name);
                if (!(v instanceof Val.Undef)) return v;
            }
            return frame.get(name);
        }

        Val evalBin(String op, Val l, Val r) {
            return switch (op) {
                case "+" -> l.isString() || r.isString()
                    ? new Val.Str(l.asStr() + r.asStr())
                    : new Val.Flt(toNum(l) + toNum(r));
                case "-" -> new Val.Flt(toNum(l) - toNum(r));
                case "*" -> new Val.Flt(toNum(l) * toNum(r));
                case "/" -> new Val.Flt(toNum(l) / toNum(r));
                case "%" -> new Val.Int((long) toNum(l) % (long) toNum(r));
                case "~" -> new Val.Str(l.asStr() + r.asStr());
                case "==" -> new Val.Bool(eq(l, r));
                case "!=" -> new Val.Bool(!eq(l, r));
                case "<"  -> new Val.Bool(toNum(l) < toNum(r));
                case ">"  -> new Val.Bool(toNum(l) > toNum(r));
                case "<=" -> new Val.Bool(toNum(l) <= toNum(r));
                case ">=" -> new Val.Bool(toNum(l) >= toNum(r));
                // and/or return an OPERAND (Python/Jinja semantics), not a coerced bool — this is
                // what makes the common `x or 'default'` / `a.get('k') or fallback` idiom work.
                case "and" -> l.truthy() ? r : l;
                case "or"  -> l.truthy() ? l : r;
                case "in" -> new Val.Bool(contains(r, l));
                case "notin" -> new Val.Bool(!contains(r, l));
                default -> Val.NONE;
            };
        }

        Val evalUnary(String op, Val arg) {
            if ("not".equals(op)) return new Val.Bool(!arg.truthy());
            return arg;
        }

        Val evalMember(Val obj, MemberNode mm) {
            if (mm.computed()) {
                if (mm.prop() instanceof SliceNode s) {
                    return slice(obj, eval(s.start()), eval(s.stop()), eval(s.step()));
                }
                Val key = eval(mm.prop());
                // integer subscript into a sequence/string (Python-style, negatives allowed)
                if ((obj instanceof Val.Arr || obj instanceof Val.Str) && isIntIndex(key))
                    return index(obj, (int) toNum(key));
                return access(obj, key.asStr());
            }
            String prop = mm.prop() instanceof LitNode l ? String.valueOf(l.val()) : eval(mm.prop()).asStr();
            return access(obj, prop);
        }

        static boolean isIntIndex(Val v) {
            return v instanceof Val.Int || (v instanceof Val.Flt f && f.v == Math.floor(f.v) && !Double.isInfinite(f.v));
        }

        /** {@code seq[i]} with Python negative indexing; out-of-range yields undefined. */
        Val index(Val obj, int i) {
            if (obj instanceof Val.Arr a) {
                int n = a.v.size(), j = i < 0 ? i + n : i;
                return j >= 0 && j < n ? a.v.get(j) : new Val.Undef("index " + i);
            }
            String str = ((Val.Str) obj).v;
            int n = str.length(), j = i < 0 ? i + n : i;
            return j >= 0 && j < n ? new Val.Str(String.valueOf(str.charAt(j))) : new Val.Undef("index " + i);
        }

        Val access(Val obj, String prop) {
            return switch (obj) {
                case Val.Obj o -> o.get(prop);
                case Val.Str s -> dispatchStrMethod(s, prop);
                case Val.Arr a -> dispatchArrMethod(a, prop);
                default -> new Val.Undef(prop);
            };
        }

        /** String methods resolve to callables so the standard {@code s.strip()} form works (and
         *  they can take arguments, e.g. {@code s.startswith(x)}, {@code s.split(',')}). */
        Val dispatchStrMethod(Val.Str s, String m) {
            return switch (m) {
                // strip/split have method-specific no-arg behavior, so they stay bespoke ...
                case "strip"  -> new Val.Func("strip", a -> new Val.Str(a.isEmpty() ? s.v.strip() : strip(s.v, expectStr(a.get(0)))));
                case "lstrip" -> new Val.Func("lstrip", a -> new Val.Str(s.v.stripLeading()));
                case "rstrip" -> new Val.Func("rstrip", a -> new Val.Str(s.v.stripTrailing()));
                case "split"  -> new Val.Func("split", a -> a.isEmpty() ? new Val.Arr(splitToValList(s.v)) : split(s, expectStr(a.get(0))));
                // ... the rest share their implementation with the identically-named filter
                case "lower", "upper", "capitalize", "startswith", "endswith", "replace" ->
                        new Val.Func(m, a -> applyFilter(m, s, a));
                case "title" -> new Val.Func("title", a -> applyFilter("capitalize", s, a));
                // unknown member: undefined (lenient, like Jinja) so `x.attr is defined` guards work
                default -> new Val.Undef(m);
            };
        }

        /** Python {@code str.strip(chars)}: trim any of the given characters from both ends. */
        static String strip(String s, String chars) {
            int a = 0, b = s.length();
            while (a < b && chars.indexOf(s.charAt(a)) >= 0) a++;
            while (b > a && chars.indexOf(s.charAt(b - 1)) >= 0) b--;
            return s.substring(a, b);
        }

        static java.util.List<Val> splitToValList(String s) {
            var parts = new ArrayList<Val>();
            for (String p : s.split("\\s+"))
                if (!p.isEmpty()) parts.add(new Val.Str(p));
            return parts;
        }

        Val dispatchArrMethod(Val.Arr a, String m) {
            return switch (m) {
                case "pop" -> a.v.isEmpty() ? Val.NONE : a.v.removeLast();
                // integer indexing is handled in evalMember; unknown members are undefined (lenient)
                default -> new Val.Undef(m);
            };
        }

        /** Python-style {@code seq[start:stop:step]} over an array or string; any bound may be
         *  omitted (None) and step may be negative (e.g. {@code [::-1]} reverses). */
        Val slice(Val v, Val s, Val sp, Val st) {
            boolean isArr = v instanceof Val.Arr;
            List<Val> src = isArr ? ((Val.Arr) v).v : null;
            String str = isArr ? null : v.asStr();
            int len = isArr ? src.size() : str.length();
            int step = !st.isDefined() || st.isNone() ? 1 : (int) toNum(st);
            if (step == 0) throw unsupported("slice with step 0");
            boolean noStart = !s.isDefined() || s.isNone();
            boolean noStop  = !sp.isDefined() || sp.isNone();
            int start = noStart ? (step > 0 ? 0 : len - 1) : sliceBound((int) toNum(s), len, step);
            int stop  = noStop  ? (step > 0 ? len : -1)     : sliceBound((int) toNum(sp), len, step);
            // single pass straight into the result (no intermediate list of boxed indices)
            var out = isArr ? new ArrayList<Val>() : null;
            var sb = isArr ? null : new StringBuilder();
            for (int i = start; step > 0 ? i < stop : i > stop; i += step) {
                if (i < 0 || i >= len) continue;
                if (isArr) out.add(src.get(i)); else sb.append(str.charAt(i));
            }
            return isArr ? new Val.Arr(out) : new Val.Str(sb.toString());
        }

        /** Normalize a slice bound (add len if negative, then clamp per Python: [0,len] for a
         *  forward step, [-1,len-1] for a backward step). */
        static int sliceBound(int i, int len, int step) {
            if (i < 0) i += len;
            return step > 0 ? Math.clamp(i, 0, len) : Math.clamp(i, -1, len - 1);
        }

        static boolean eq(Val a, Val b) {
            if (a instanceof Val.Str sa && b instanceof Val.Str sb) return sa.v.equals(sb.v);
            if (a instanceof Val.Int ia && b instanceof Val.Int ib) return ia.v == ib.v;
            if (a instanceof Val.Bool ba && b instanceof Val.Bool bb) return ba.v == bb.v;
            if (a instanceof Val.None && b instanceof Val.None) return true;
            if (a instanceof Val.None || b instanceof Val.None) return false;
            return a.asStr().equals(b.asStr());
        }

        static boolean contains(Val container, Val item) {
            if (container instanceof Val.Str s) return s.v.contains(item.asStr());
            if (container instanceof Val.Arr a) {
                for (Val v : a.v()) if (eq(v, item)) return true;
            }
            if (container instanceof Val.Obj o) return o.v.containsKey(item.asStr());
            return false;
        }

        static double toNum(Val v) {
            return switch (v) {
                case Val.Int i -> (double) i.v;
                case Val.Flt f -> f.v;
                case Val.Bool b -> b.v ? 1.0 : 0.0;
                default -> {
                    try { yield Double.parseDouble(v.asStr()); }
                    catch (NumberFormatException __) { yield 0.0; }
                }
            };
        }
    }
}
