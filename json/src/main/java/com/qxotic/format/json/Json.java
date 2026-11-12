package com.qxotic.format.json;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/**
 * Minimal JSON parser and printer, RFC 8259 compliant.
 *
 * <p>JSON values map to plain Java types: objects to {@code Map<String, Object>}, arrays to {@code
 * List<Object>}, strings to {@code String}, numbers to {@code Long}, {@code BigInteger}, {@code
 * Double} or {@code BigDecimal}, booleans to {@code Boolean}, and {@code null} to the {@link #NULL}
 * singleton. {@code stringify} accepts compatible {@code Map}/{@code List} trees and rejects Java
 * {@code null}; use {@link #NULL} for JSON null.
 *
 * <pre>{@code
 * Map<String, Object> config = Json.parseMap("{\"name\": \"qxotic\", \"retries\": 3}");
 * Optional<Number> retries = Json.queryNumber(config, "retries");
 * String json = Json.stringify(config, true);
 * }</pre>
 */
public final class Json {

    private Json() {}

    /** Singleton for JSON null, used instead of Java {@code null} by both parsing and printing. */
    public static final Null NULL = Null.INSTANCE;

    /** Type of the {@link Json#NULL} singleton. */
    public static final class Null {
        private static final Null INSTANCE = new Null();

        private Null() {}

        @Override
        public String toString() {
            return "null";
        }
    }

    // === Public API ===

    /** Parse JSON into Java values using default options. */
    public static Object parse(CharSequence json) {
        return parse(json, ParseOptions.defaults());
    }

    /** Parse JSON into Java values with custom options. */
    public static Object parse(CharSequence json, ParseOptions options) {
        Objects.requireNonNull(json, "json");
        Objects.requireNonNull(options, "options");
        Parser p = new Parser(json, options);
        Object result = p.parseValue();
        p.expectEnd();
        return result;
    }

    /** Return true when input is valid JSON with default options. */
    public static boolean isValid(CharSequence json) {
        return isValid(json, ParseOptions.defaults());
    }

    /** Return true when input is valid JSON for the given options. */
    public static boolean isValid(CharSequence json, ParseOptions options) {
        Objects.requireNonNull(json, "json");
        Objects.requireNonNull(options, "options");
        try {
            Parser p = new Parser(json, options);
            p.skipValue();
            p.expectEnd();
            return true;
        } catch (ParseException e) {
            return false;
        }
    }

    /** Parse JSON and require a Map (object) root. */
    public static Map<String, Object> parseMap(CharSequence json) {
        return parseMap(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require a Map (object) root. */
    public static Map<String, Object> parseMap(CharSequence json, ParseOptions options) {
        return castMap(requireRootType(parse(json, options), Map.class, "object"));
    }

    /** Parse JSON and require a List (array) root. */
    public static List<Object> parseList(CharSequence json) {
        return parseList(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require a List (array) root. */
    public static List<Object> parseList(CharSequence json, ParseOptions options) {
        return castList(requireRootType(parse(json, options), List.class, "array"));
    }

    /** Parse JSON and require a string root. */
    public static String parseString(CharSequence json) {
        return (String)
                requireRootType(parse(json, ParseOptions.defaults()), String.class, "string");
    }

    /** Parse JSON and require a number root. */
    public static Number parseNumber(CharSequence json) {
        return parseNumber(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require a number root. */
    public static Number parseNumber(CharSequence json, ParseOptions options) {
        return (Number) requireRootType(parse(json, options), Number.class, "number");
    }

    /** Parse JSON and require a boolean root. */
    public static boolean parseBoolean(CharSequence json) {
        return (Boolean)
                requireRootType(parse(json, ParseOptions.defaults()), Boolean.class, "boolean");
    }

    /** Serialize Java value to compact JSON. */
    public static String stringify(Object value) {
        return stringify(value, false);
    }

    /** Serialize Java value to JSON, optionally pretty-printed. */
    public static String stringify(Object value, boolean pretty) {
        StringBuilder sb = new StringBuilder(estimateSize(value));
        print(sb, value, pretty, 0, null);
        return sb.toString();
    }

    /**
     * Escape a raw Java string for use inside a JSON string value. Does not add surrounding quotes.
     */
    public static String escapeString(CharSequence s) {
        Objects.requireNonNull(s, "s");
        int i = 0;
        while (i < s.length() && plain(s.charAt(i))) {
            i++;
        }
        if (i == s.length()) {
            return s.toString(); // nothing to escape
        }
        StringBuilder sb = new StringBuilder(s.length() + 16);
        appendEscaped(sb, s);
        return sb.toString();
    }

    /**
     * Unescape a JSON-escaped string back to raw Java text. Does not remove surrounding quotes.
     *
     * @throws ParseException on a malformed escape, as {@link #parse} would
     */
    public static String unescapeString(CharSequence s) {
        Objects.requireNonNull(s, "s");
        return new Parser(s, ParseOptions.defaults()).unescapeAll();
    }

    /** A first capacity for the output: a small document then needs no regrowth (measured). */
    private static int estimateSize(Object value) {
        if (value instanceof Map) {
            return Math.max(64, ((Map<?, ?>) value).size() * 32);
        }
        if (value instanceof List) {
            return Math.max(64, ((List<?>) value).size() * 16);
        }
        return 32;
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castMap(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> castList(Object value) {
        return (List<Object>) value;
    }

    private static Object requireRootType(Object value, Class<?> rootType, String typeName) {
        if (!rootType.isInstance(value)) {
            throw new ParseException("Expected JSON " + typeName + " at root");
        }
        return value;
    }

    // === Parser ===

    private static final class Parser {
        private final CharSequence input;
        private final ParseOptions options;
        private int pos;
        private int depth;
        private final Map<String, String> stringInterner = new HashMap<>();

        Parser(CharSequence input, ParseOptions options) {
            this.input = input;
            this.options = options;
        }

        /** The whole input as a string's content: its escapes decoded, everything else kept. */
        private String unescapeAll() {
            if (input instanceof String) {
                pos =
                        ((String) input)
                                .indexOf('\\'); // String's own search, much faster than a loop
                if (pos < 0) {
                    return (String) input;
                }
            }
            StringBuilder sb = null;
            int start = 0;
            while (pos < input.length()) {
                if (input.charAt(pos) != '\\') {
                    pos++;
                    continue;
                }
                if (sb == null) {
                    sb = new StringBuilder(input.length());
                }
                sb.append(input, start, pos);
                if (++pos == input.length()) {
                    throw errorAt(pos - 1, "Invalid escape sequence", null); // nothing after it
                }
                appendEscapedChar(sb); // leaves pos past the escape
                start = pos;
            }
            return sb == null
                    ? input.toString()
                    : sb.append(input, start, input.length()).toString();
        }

        private String intern(String s) {
            String existing = stringInterner.putIfAbsent(s, s);
            return existing != null ? existing : s;
        }

        private Object parseValue() {
            skipSpace();
            char ch = peek();

            switch (ch) {
                case '"':
                    return parseString();
                case '[':
                    next();
                    return parseArray();
                case '{':
                    next();
                    return parseObject();
                case 't':
                    expect("true");
                    return true;
                case 'f':
                    expect("false");
                    return false;
                case 'n':
                    expect("null");
                    return NULL;
                case ']':
                    throw error("Expected value");
                case '}':
                case ',':
                    throw error("Unexpected character");
                default:
                    if (ch == '-' || isDigit(ch)) {
                        return parseNumber();
                    }
                    throw error("Unexpected character");
            }
        }

        /** Validate JSON structure without allocating parsed objects. */
        private void skipValue() {
            skipSpace();
            char ch = peek();

            switch (ch) {
                case '"':
                    skipString();
                    break;
                case '[':
                    next();
                    skipArray();
                    break;
                case '{':
                    next();
                    skipObject();
                    break;
                case 't':
                    expect("true");
                    break;
                case 'f':
                    expect("false");
                    break;
                case 'n':
                    expect("null");
                    break;
                case ']':
                    throw error("Expected value");
                case '}':
                case ',':
                    throw error("Unexpected character");
                default:
                    if (ch == '-' || isDigit(ch)) {
                        scanNumber();
                    } else {
                        throw error("Unexpected character");
                    }
                    break;
            }
        }

        private void skipString() {
            expect('"');
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (ch == '"') {
                    pos++;
                    return;
                }
                if (ch == '\\') {
                    pos++;
                    skipEscapeSequence();
                } else if (ch < 0x20) {
                    throw error("Control character must be escaped");
                } else {
                    pos++;
                }
            }
            throw error("Unexpected end of input");
        }

        private void skipEscapeSequence() {
            char ch = next();
            switch (ch) {
                case '"':
                case '\\':
                case '/':
                case 'b':
                case 'f':
                case 'n':
                case 'r':
                case 't':
                    break;
                case 'u':
                    validateUnicodeEscape();
                    break;
                default:
                    throw errorAt(pos - 2, "Invalid escape sequence", null); // at its backslash
            }
        }

        /**
         * Scan and validate a JSON number, advancing pos. Returns true if it has a decimal part.
         */
        private boolean scanNumber() {
            // Sign
            if (peek() == '-') {
                next();
            } else if (peek() == '+') {
                throw error("Unexpected '+'");
            }

            // Integer part
            if (!isDigit(peek())) {
                if (peek() == '.') {
                    throw error("Unexpected '.'");
                }
                throw error("Expected digit");
            }

            // No leading zeros (except single zero)
            if (peek() == '0' && pos + 1 < input.length() && isDigit(input.charAt(pos + 1))) {
                throw error("Leading zeros not allowed");
            }

            while (isDigit(peek())) {
                next();
            }

            boolean hasDecimal = false;

            // Fraction
            if (peek() == '.') {
                hasDecimal = true;
                next();
                if (!isDigit(peek())) {
                    throw error("Expected digit after decimal point");
                }
                while (isDigit(peek())) {
                    next();
                }
            }

            // Exponent
            if (peek() == 'e' || peek() == 'E') {
                hasDecimal = true;
                next();
                if (peek() == '+' || peek() == '-') {
                    next();
                }
                if (!isDigit(peek())) {
                    throw error("Exponent missing digits");
                }
                while (isDigit(peek())) {
                    next();
                }
            }

            return hasDecimal;
        }

        private void skipArray() {
            enterDepth();
            try {
                skipSpace();
                if (peek() == ']') {
                    next();
                    return;
                }
                while (true) {
                    skipValue();
                    if (!consumeCommaOrEnd(']')) {
                        break;
                    }
                }
            } finally {
                exitDepth();
            }
        }

        private void skipObject() {
            enterDepth();
            try {
                skipSpace();
                if (peek() == '}') {
                    next();
                    return;
                }
                Set<String> keys = options.failOnDuplicateKeys() ? new HashSet<>() : null;
                while (true) {
                    if (keys != null) {
                        int at = pos;
                        String key = parseString();
                        if (!keys.add(key)) {
                            throw errorAt(at, "Duplicate key: '" + key + "'", null);
                        }
                    } else {
                        skipString();
                    }
                    skipSpace();
                    expect(':');
                    skipValue();
                    if (!consumeCommaOrEnd('}')) {
                        break;
                    }
                }
            } finally {
                exitDepth();
            }
        }

        private String parseString() {
            expect('"');
            int start = pos;

            // Fast path: scan for end of string or escape sequence
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (ch == '"') {
                    // No escapes found - use substring directly
                    String result = lexeme(start);
                    pos++; // Skip closing quote
                    return result;
                }
                if (ch == '\\') {
                    // Escapes found - use StringBuilder
                    break;
                }
                if (ch < 0x20) {
                    throw error("Control character must be escaped");
                }
                pos++;
            }

            // Slow path: has escape sequences
            return parseStringWithEscapes(start);
        }

        private String parseStringWithEscapes(int start) {
            StringBuilder sb = new StringBuilder();

            // Copy chars before first escape
            sb.append(input, start, pos);

            // Process escapes
            while (true) {
                char ch = next();
                if (ch == '"') {
                    return sb.toString();
                }
                if (ch == '\\') {
                    appendEscapedChar(sb);
                } else if (ch < 0x20) {
                    throw error("Control character must be escaped");
                } else {
                    sb.append(ch);
                }
            }
        }

        private void appendEscapedChar(StringBuilder sb) {
            char ch = next();
            switch (ch) {
                case '"':
                    sb.append('"');
                    break;
                case '\\':
                    sb.append('\\');
                    break;
                case '/':
                    sb.append('/');
                    break;
                case 'b':
                    sb.append('\b');
                    break;
                case 'f':
                    sb.append('\f');
                    break;
                case 'n':
                    sb.append('\n');
                    break;
                case 'r':
                    sb.append('\r');
                    break;
                case 't':
                    sb.append('\t');
                    break;
                case 'u':
                    sb.appendCodePoint(validateUnicodeEscape());
                    break;
                default:
                    throw errorAt(pos - 2, "Invalid escape sequence", null); // at its backslash
            }
        }

        /**
         * Validate and advance past a \\uXXXX (possibly surrogate pair). Returns the code point.
         */
        private int validateUnicodeEscape() {
            int code = parseHex4();

            if (Character.isHighSurrogate((char) code)) {
                if (pos + 6 > input.length()
                        || input.charAt(pos) != '\\'
                        || input.charAt(pos + 1) != 'u') {
                    throw error("Lone surrogate");
                }
                pos += 2; // Skip backslash-u
                int low = parseHex4();
                if (!Character.isLowSurrogate((char) low)) {
                    throw error("Unexpected character after high surrogate");
                }
                return Character.toCodePoint((char) code, (char) low);
            } else if (Character.isLowSurrogate((char) code)) {
                throw error("Lone surrogate");
            }
            return code;
        }

        private int parseHex4() {
            if (pos + 4 > input.length()) {
                throw error("Incomplete Unicode escape");
            }

            int result = 0;
            for (int i = 0; i < 4; i++) {
                char ch = next();
                int digit = Character.digit(ch, 16);
                if (digit < 0) {
                    throw error("Invalid hex digit");
                }
                result = (result << 4) | digit;
            }
            return result;
        }

        private List<Object> parseArray() {
            enterDepth();
            try {
                List<Object> list = new ArrayList<>();

                skipSpace();
                if (peek() == ']') {
                    next();
                    return list;
                }

                while (true) {
                    list.add(parseValue());
                    if (!consumeCommaOrEnd(']')) {
                        break;
                    }
                }
                return list;
            } finally {
                exitDepth();
            }
        }

        private Map<String, Object> parseObject() {
            enterDepth();
            try {
                Map<String, Object> map = new LinkedHashMap<>(16);

                skipSpace();
                if (peek() == '}') {
                    next();
                    return map;
                }

                while (true) {
                    int at = pos;
                    String key = intern(parseString());
                    if (options.failOnDuplicateKeys() && map.containsKey(key)) {
                        throw errorAt(at, "Duplicate key: '" + key + "'", null);
                    }
                    skipSpace();
                    expect(':');
                    map.put(key, parseValue());
                    if (!consumeCommaOrEnd('}')) {
                        break;
                    }
                }
                return map;
            } finally {
                exitDepth();
            }
        }

        private static final long LONG_OVERFLOW_THRESHOLD = Long.MAX_VALUE / 10;

        private Number parseNumber() {
            int start = pos;
            boolean hasDecimal = scanNumber();

            if (hasDecimal) {
                String lexeme = lexeme(start);
                if (options.decimalsAsBigDecimal()) {
                    try {
                        return new BigDecimal(lexeme);
                    } catch (NumberFormatException e) {
                        throw errorAt(pos, "Invalid number", e);
                    }
                }
                double d = Double.parseDouble(lexeme);
                if (Double.isInfinite(d)) {
                    throw errorAt(start, "Number out of double range", null);
                }
                return d;
            }

            // Integer - inline Long accumulation over validated digits
            int i = start;
            boolean negative = input.charAt(i) == '-';
            if (negative) i++;

            long value = 0;
            while (i < pos) {
                int digit = input.charAt(i++) - '0';
                if (value > LONG_OVERFLOW_THRESHOLD
                        || (value == LONG_OVERFLOW_THRESHOLD && digit > (negative ? 8 : 7))) {
                    return new BigInteger(lexeme(start));
                }
                value = value * 10 + digit;
            }
            return negative ? -value : value;
        }

        // === Utilities ===

        private String lexeme(int start) {
            return input instanceof String
                    ? ((String) input).substring(start, pos)
                    : input.subSequence(start, pos).toString();
        }

        private void skipSpace() {
            while (pos < input.length() && isWhitespace(input.charAt(pos))) {
                pos++;
            }
        }

        private char peek() {
            return pos < input.length() ? input.charAt(pos) : '\0';
        }

        private char next() {
            if (pos >= input.length()) {
                throw error("Unexpected end of input");
            }
            return input.charAt(pos++);
        }

        private void expect(char ch) {
            if (pos >= input.length() || input.charAt(pos) != ch) {
                throw error("Expected '" + ch + "'");
            }
            pos++;
        }

        private void expect(String str) {
            for (int i = 0; i < str.length(); i++) {
                if (pos >= input.length() || input.charAt(pos) != str.charAt(i)) {
                    throw error("Expected '" + str + "'");
                }
                pos++;
            }
        }

        private void expectEnd() {
            skipSpace();
            if (pos < input.length()) throw error("Expected end of input");
        }

        private boolean consumeCommaOrEnd(char end) {
            skipSpace();
            char ch = peek();
            if (ch == ',') {
                next();
                skipSpace();
                if (peek() == end) {
                    throw error("Expected value");
                }
                return true;
            }
            if (ch == end) {
                next();
                return false;
            }
            if (ch == '\0') {
                throw error("Expected '" + end + "'");
            }
            throw error("Expected ',' or '" + end + "'");
        }

        private void enterDepth() {
            if (++depth > options.maxDepth()) {
                throw error("Maximum parsing depth exceeded");
            }
        }

        private void exitDepth() {
            depth--;
        }

        private ParseException error(String msg) {
            return errorAt(pos, msg, null);
        }

        private ParseException errorAt(int at, String msg, Throwable cause) {
            int[] lc = lineColumnAt(at);
            return new ParseException(msg, at, lc[0], lc[1], input, cause);
        }

        /** Line and column of {@code index}: a line ends at \n, \r or \r\n alike. */
        private int[] lineColumnAt(int index) {
            int line = 1;
            int column = 1;
            int end = Math.min(index, input.length());
            for (int i = 0; i < end; i++) {
                char ch = input.charAt(i);
                boolean lineEnd =
                        ch == '\n'
                                || (ch == '\r'
                                        && (i + 1 >= input.length()
                                                || input.charAt(i + 1) != '\n'));
                if (lineEnd) {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
            }
            return new int[] {line, column};
        }

        private static boolean isDigit(char ch) {
            return ch >= '0' && ch <= '9';
        }

        private static boolean isWhitespace(char ch) {
            return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
        }
    }

    // === Printer ===

    private static String escapeFor(char ch) {
        switch (ch) {
            case '"':
                return "\\\"";
            case '\\':
                return "\\\\";
            case '\b':
                return "\\b";
            case '\f':
                return "\\f";
            case '\n':
                return "\\n";
            case '\r':
                return "\\r";
            case '\t':
                return "\\t";
            default:
                return ch < 0x20 ? unicodeEscape(ch) : null;
        }
    }

    private static String unicodeEscape(int cp) {
        String hex = Integer.toHexString(cp);
        return "\\u" + "0000".substring(hex.length()) + hex.toUpperCase();
    }

    private static void print(
            StringBuilder sb, Object value, boolean pretty, int indent, Set<Object> visiting) {
        if (value == NULL) {
            sb.append("null");
            return;
        }
        if (value == null) {
            throw new IllegalArgumentException(
                    "Cannot serialize Java null; use Json.NULL for JSON null");
        }
        if (value instanceof Boolean) {
            sb.append(value);
            return;
        }
        if (value instanceof Number) {
            printNumber(sb, (Number) value);
            return;
        }
        if (value instanceof CharSequence) {
            printString(sb, (CharSequence) value);
            return;
        }
        if (value instanceof List) {
            printList(sb, (List<?>) value, pretty, indent, visiting);
            return;
        }
        if (value instanceof Map) {
            printMap(sb, (Map<?, ?>) value, pretty, indent, visiting);
            return;
        }
        throw new IllegalArgumentException("Cannot serialize: " + value.getClass());
    }

    private static void printNumber(StringBuilder sb, Number num) {
        if (num instanceof Double || num instanceof Float) {
            double d = num.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                throw new IllegalArgumentException("Cannot serialize NaN/Infinity");
            }
            if (d == Math.rint(d) && Math.abs(d) < 0x1p53) {
                // integral and below 2^53, where a double still holds every integer: printed as
                // digits it reads back exactly; above that the digits would claim a precision
                // the value never had (and the cast to long saturates past 2^63)
                sb.append((long) d);
            } else if (num instanceof Float) {
                sb.append(num.floatValue()); // Float.toString: 0.1f prints 0.1, not 0.1000000015
            } else {
                sb.append(d);
            }
        } else if (num instanceof BigDecimal) {
            BigDecimal bd = (BigDecimal) num;
            sb.append(bd.stripTrailingZeros().toPlainString());
        } else {
            sb.append(num);
        }
    }

    private static void printString(StringBuilder sb, CharSequence str) {
        sb.append('"');
        appendEscaped(sb, str);
        sb.append('"');
    }

    /**
     * Whether {@code ch} is written as itself: not a quote, a backslash, a control or a surrogate.
     */
    private static boolean plain(char ch) {
        return ch >= 0x20 && ch != '"' && ch != '\\' && !Character.isSurrogate(ch);
    }

    /** {@code s} as a JSON string's content. */
    private static void appendEscaped(StringBuilder sb, CharSequence s) {
        int start = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (plain(ch)) {
                continue;
            }
            if (Character.isHighSurrogate(ch)
                    && i + 1 < s.length()
                    && Character.isLowSurrogate(s.charAt(i + 1))) {
                i++; // a surrogate pair, written as it is
                continue;
            }
            // a lone surrogate has no UTF-8 form: escaped, the text stays encodable
            String escape = Character.isSurrogate(ch) ? unicodeEscape(ch) : escapeFor(ch);
            if (escape != null) {
                sb.append(s, start, i).append(escape);
                start = i + 1;
            }
        }
        sb.append(s, start, s.length());
    }

    private static void printList(
            StringBuilder sb, List<?> list, boolean pretty, int indent, Set<Object> visiting) {
        visiting = enterContainer(list, visiting);
        try {
            sb.append('[');
            boolean first = true;
            for (Object elem : list) {
                if (!first) {
                    sb.append(',');
                }
                appendPrettyIndent(sb, pretty, indent + 1);
                print(sb, elem, pretty, indent + 1, visiting);
                first = false;
            }
            appendPrettyIndent(sb, pretty && !first, indent);
            sb.append(']');
        } finally {
            exitContainer(list, visiting);
        }
    }

    private static void printMap(
            StringBuilder sb, Map<?, ?> map, boolean pretty, int indent, Set<Object> visiting) {
        visiting = enterContainer(map, visiting);
        try {
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (!first) {
                    sb.append(',');
                }
                appendPrettyIndent(sb, pretty, indent + 1);
                Object key = entry.getKey();
                if (!(key instanceof CharSequence)) {
                    throw new IllegalArgumentException(
                            "JSON object keys must be strings, got: "
                                    + (key == null ? "null" : key.getClass().getName()));
                }
                printString(sb, (CharSequence) key);
                sb.append(pretty ? ": " : ":");
                print(sb, entry.getValue(), pretty, indent + 1, visiting);
                first = false;
            }
            appendPrettyIndent(sb, pretty && !first, indent);
            sb.append('}');
        } finally {
            exitContainer(map, visiting);
        }
    }

    private static Set<Object> enterContainer(Object container, Set<Object> visiting) {
        if (visiting == null) {
            visiting = Collections.newSetFromMap(new IdentityHashMap<>());
        }
        if (!visiting.add(container)) {
            throw new IllegalArgumentException("Cannot serialize cyclic structure");
        }
        return visiting;
    }

    private static void exitContainer(Object container, Set<Object> visiting) {
        visiting.remove(container);
    }

    private static void appendPrettyIndent(StringBuilder sb, boolean enabled, int indent) {
        if (enabled) {
            sb.append('\n');
            for (int i = 0; i < indent; i++) {
                sb.append("  ");
            }
        }
    }

    // === ParseOptions ===

    /**
     * Immutable parsing options. Each wither method returns a modified copy; {@link #defaults()}
     * uses {@link #DEFAULT_MAX_DEPTH} and parses decimals as {@code BigDecimal}.
     */
    public static final class ParseOptions {
        /** Default maximum object/array nesting depth. */
        public static final int DEFAULT_MAX_DEPTH = 1000;

        private final boolean decimalsAsBigDecimal;
        private final int maxDepth;
        private final boolean failOnDuplicateKeys;

        private ParseOptions(
                boolean decimalsAsBigDecimal, int maxDepth, boolean failOnDuplicateKeys) {
            this.decimalsAsBigDecimal = decimalsAsBigDecimal;
            this.maxDepth = maxDepth;
            this.failOnDuplicateKeys = failOnDuplicateKeys;
        }

        /** Create default parse options. */
        public static ParseOptions defaults() {
            return new ParseOptions(true, DEFAULT_MAX_DEPTH, false);
        }

        /**
         * Return new options with {@code decimalsAsBigDecimal} set. When true, decimals parse as
         * {@code BigDecimal}; when false, as {@code Double}.
         */
        public ParseOptions decimalsAsBigDecimal(boolean enabled) {
            return new ParseOptions(enabled, this.maxDepth, this.failOnDuplicateKeys);
        }

        /** Return new options with {@code maxDepth} set. Must be positive. */
        public ParseOptions maxDepth(int depth) {
            if (depth <= 0) {
                throw new IllegalArgumentException("Maximum parsing depth must be positive");
            }
            return new ParseOptions(this.decimalsAsBigDecimal, depth, this.failOnDuplicateKeys);
        }

        /** Return new options with {@code failOnDuplicateKeys} set. */
        public ParseOptions failOnDuplicateKeys(boolean enabled) {
            return new ParseOptions(this.decimalsAsBigDecimal, this.maxDepth, enabled);
        }

        public boolean decimalsAsBigDecimal() {
            return decimalsAsBigDecimal;
        }

        public int maxDepth() {
            return maxDepth;
        }

        public boolean failOnDuplicateKeys() {
            return failOnDuplicateKeys;
        }
    }

    // === ParseException ===

    /**
     * Thrown when JSON input is malformed. Exceptions produced by the parser include line, column,
     * and a caret pointing at the offending position.
     */
    public static final class ParseException extends RuntimeException {
        private static final long serialVersionUID = 1L;

        private final int position;
        private final int line;
        private final int column;

        public ParseException(String message) {
            super(message);
            this.position = -1;
            this.line = -1;
            this.column = -1;
        }

        public ParseException(String message, Throwable cause) {
            super(message, cause);
            this.position = -1;
            this.line = -1;
            this.column = -1;
        }

        private ParseException(
                String message,
                int position,
                int line,
                int column,
                CharSequence input,
                Throwable cause) {
            super(formatMessage(message, line, column, input, position), cause);
            this.position = position;
            this.line = line;
            this.column = column;
        }

        public int getPosition() {
            return position;
        }

        public int getLine() {
            return line;
        }

        public int getColumn() {
            return column;
        }

        private static String formatMessage(
                String message, int line, int column, CharSequence input, int position) {
            String base = "Line " + line + ", Column " + column + ": " + message;
            if (input == null || position < 0) {
                return base;
            }

            int start = position;
            while (start > 0) {
                char ch = input.charAt(start - 1);
                if (ch == '\n' || ch == '\r') {
                    break;
                }
                start--;
            }

            int end = position;
            while (end < input.length()) {
                char ch = input.charAt(end);
                if (ch == '\n' || ch == '\r') {
                    break;
                }
                end++;
            }

            String lineText = input.subSequence(start, end).toString();
            int caret = Math.max(0, Math.min(position - start, lineText.length()));
            return base + "\n" + lineText + "\n" + " ".repeat(caret) + "^";
        }
    }

    // === Query Methods - Navigate through Map keys with varargs ===

    /**
     * Query any value by navigating through object keys. Returns Optional.of(Json.NULL) for
     * explicit null values. Returns empty if path doesn't exist.
     */
    public static Optional<Object> query(Object root, String... keys) {
        Object value = navigate(root, keys);
        return value != null ? Optional.of(value) : Optional.empty();
    }

    /** Query a String value by navigating through object keys. */
    public static Optional<String> queryString(Object root, String... keys) {
        return query(root, keys).filter(v -> v instanceof String).map(v -> (String) v);
    }

    /** Query a Map value by navigating through object keys. */
    @SuppressWarnings("unchecked")
    public static Optional<Map<String, Object>> queryMap(Object root, String... keys) {
        return query(root, keys).filter(v -> v instanceof Map).map(v -> (Map<String, Object>) v);
    }

    /** Query a List value by navigating through object keys. */
    @SuppressWarnings("unchecked")
    public static Optional<List<Object>> queryList(Object root, String... keys) {
        return query(root, keys).filter(v -> v instanceof List).map(v -> (List<Object>) v);
    }

    /** Query a Boolean value by navigating through object keys. */
    public static Optional<Boolean> queryBoolean(Object root, String... keys) {
        return query(root, keys).filter(v -> v instanceof Boolean).map(v -> (Boolean) v);
    }

    /** Query a Number value by navigating through object keys. */
    public static Optional<Number> queryNumber(Object root, String... keys) {
        return query(root, keys).filter(v -> v instanceof Number).map(v -> (Number) v);
    }

    /**
     * Navigate through nested Maps using the provided keys. Returns null if navigation fails
     * (missing key, wrong type, etc.) Returns Json.NULL for explicit null values. With 0 keys,
     * returns root directly (acts as cast).
     */
    private static Object navigate(Object root, String... keys) {
        Objects.requireNonNull(root, "root");
        if (keys == null || keys.length == 0) {
            return root;
        }

        Object current = root;
        for (String key : keys) {
            Objects.requireNonNull(key, "key");
            if (!(current instanceof Map)) {
                return null;
            }
            @SuppressWarnings("unchecked")
            Map<String, ?> map = (Map<String, ?>) current;
            current = map.get(key);
            if (current == null) {
                return null;
            }
        }
        return current;
    }
}
