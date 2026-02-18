package ai.qxotic.format.json;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Collections;
import java.util.IdentityHashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/** Beautiful, minimal JSON parser and printer. RFC 8259 compliant, elegant, and fast. */
public final class JSON {

    /** JSON null distinct from Java null. */
    public static final Object NULL =
            new Object() {
                @Override
                public String toString() {
                    return "null";
                }
            };

    // === Public API ===

    /** Parse JSON string into Java objects. */
    public static Object parse(CharSequence json) {
        return parse(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options. */
    public static Object parse(CharSequence json, ParseOptions options) {
        requireJsonInput(json);
        if (options == null) {
            throw new IllegalArgumentException("options must not be null");
        }
        Parser p = new Parser(json, options);
        Object result = p.parseValue();
        p.expectEnd();
        return result;
    }

    /** Returns true if input is valid JSON. */
    public static boolean isValid(CharSequence json) {
        if (json == null) {
            return false;
        }
        try {
            parse(json);
            return true;
        } catch (ParseException e) {
            return false;
        }
    }

    /** Parse JSON and require an object root. */
    public static Map<String, Object> parseObject(CharSequence json) {
        return parseObject(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require an object root. */
    public static Map<String, Object> parseObject(CharSequence json, ParseOptions options) {
        return castObject(requireRootType(parse(json, options), Map.class, "object"));
    }

    /** Parse JSON and require an array root. */
    public static List<Object> parseArray(CharSequence json) {
        return parseArray(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require an array root. */
    public static List<Object> parseArray(CharSequence json, ParseOptions options) {
        return castArray(requireRootType(parse(json, options), List.class, "array"));
    }

    /** Parse JSON and require a string root. */
    public static String parseString(CharSequence json) {
        return parseString(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require a string root. */
    public static String parseString(CharSequence json, ParseOptions options) {
        return (String) requireRootType(parse(json, options), String.class, "string");
    }

    /** Parse JSON and require a number root. */
    public static Number parseNumber(CharSequence json) {
        return parseNumber(json, ParseOptions.defaults());
    }

    /** Parse JSON with custom options and require a number root. */
    public static Number parseNumber(CharSequence json, ParseOptions options) {
        return (Number) requireRootType(parse(json, options), Number.class, "number");
    }

    /** Convert Java object to compact JSON string. */
    public static String stringify(Object value) {
        return stringify(value, false);
    }

    /** Convert Java object to pretty JSON string. */
    public static String stringifyPretty(Object value) {
        return stringify(value, true);
    }

    /** Convert to JSON with optional pretty-printing. */
    public static String stringify(Object value, boolean pretty) {
        StringBuilder sb = new StringBuilder();
        Set<Object> visiting = Collections.newSetFromMap(new IdentityHashMap<>());
        print(sb, value, pretty, 0, visiting);
        return sb.toString();
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> castObject(Object value) {
        return (Map<String, Object>) value;
    }

    @SuppressWarnings("unchecked")
    private static List<Object> castArray(Object value) {
        return (List<Object>) value;
    }

    private static void requireJsonInput(CharSequence json) {
        if (json == null) {
            throw new IllegalArgumentException("json must not be null");
        }
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

        private static final class LineColumn {
            final int line;
            final int column;

            LineColumn(int line, int column) {
                this.line = line;
                this.column = column;
            }
        }

        Parser(CharSequence input, ParseOptions options) {
            this.input = input;
            this.options = options;
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

        private String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();

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
                    appendUnicodeEscape(sb);
                    break;
                default:
                    throw error("Invalid escape sequence");
            }
        }

        private void appendUnicodeEscape(StringBuilder sb) {
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
                sb.append((char) code).append((char) low);
            } else {
                if (Character.isLowSurrogate((char) code)) {
                    throw error("Lone surrogate");
                }
                sb.append((char) code);
            }
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
                Map<String, Object> map = new LinkedHashMap<>();

                skipSpace();
                if (peek() == '}') {
                    next();
                    return map;
                }

                while (true) {
                    String key = parseString();
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

        private Number parseNumber() {
            int start = pos;
            boolean hasFraction = false;
            boolean hasExponent = false;

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

            // Fraction
            if (peek() == '.') {
                hasFraction = true;
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
                hasExponent = true;
                next();
                if (peek() == '+' || peek() == '-') {
                    next();
                }
                if (!isDigit(peek())) {
                    throw error("exponent missing digits");
                }
                while (isDigit(peek())) {
                    next();
                }
            }

            String numberLexeme = input.subSequence(start, pos).toString();
            boolean isDecimal = hasFraction || hasExponent;

            if (!isDecimal) {
                if (numberLexeme.equals("-0")) {
                    return new BigDecimal("-0");
                }
                try {
                    return Long.parseLong(numberLexeme);
                } catch (NumberFormatException e) {
                    return new BigInteger(numberLexeme);
                }
            }

            if (options.shouldUseBigDecimal()) {
                return parseBigDecimal(numberLexeme);
            }

            try {
                return Double.parseDouble(numberLexeme);
            } catch (NumberFormatException e) {
                throw error("Invalid number", e);
            }
        }

        private Number parseBigDecimal(String numberLexeme) {
            try {
                BigDecimal bd = new BigDecimal(numberLexeme);
                return bd.compareTo(BigDecimal.ZERO) == 0 ? bd.stripTrailingZeros() : bd;
            } catch (NumberFormatException e) {
                throw error("Invalid number", e);
            }
        }

        // === Utilities ===

        private void skipSpace() {
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (isWhitespace(ch)) {
                    pos++;
                } else {
                    break;
                }
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
            if (pos >= input.length()) {
                throw error("Expected '" + ch + "'");
            }
            if (next() != ch) {
                throw error("Expected '" + ch + "'");
            }
        }

        private void expect(String str) {
            for (int i = 0; i < str.length(); i++) {
                if (pos >= input.length()) {
                    throw error("Expected '" + str + "'");
                }
                if (next() != str.charAt(i)) {
                    throw error("Expected '" + str + "'");
                }
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
            if (++depth > options.getMaxParsingDepth()) {
                throw error("Maximum parsing depth exceeded");
            }
        }

        private void exitDepth() {
            depth--;
        }

        private ParseException error(String msg) {
            LineColumn lc = lineColumnAt(pos);
            return new ParseException(msg, pos, lc.line, lc.column, input);
        }

        private ParseException error(String msg, Throwable cause) {
            LineColumn lc = lineColumnAt(pos);
            return new ParseException(msg, pos, lc.line, lc.column, input, cause);
        }

        private LineColumn lineColumnAt(int index) {
            int line = 1;
            int column = 1;
            int end = Math.min(index, input.length());
            for (int i = 0; i < end; i++) {
                char ch = input.charAt(i);
                if (ch == '\n') {
                    line++;
                    column = 1;
                } else {
                    column++;
                }
            }
            return new LineColumn(line, column);
        }

        private static boolean isDigit(char ch) {
            return ch >= '0' && ch <= '9';
        }

        private static boolean isWhitespace(char ch) {
            return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
        }
    }

    // === Printer ===

    private static void print(
            StringBuilder sb, Object value, boolean pretty, int indent, Set<Object> visiting) {
        if (value == null || value == NULL) {
            sb.append("null");
            return;
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
            printArray(sb, (List<?>) value, pretty, indent, visiting);
            return;
        }
        if (value instanceof Map) {
            printObject(sb, (Map<?, ?>) value, pretty, indent, visiting);
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
            if (d == (long) d) {
                sb.append((long) d);
            } else {
                sb.append(d);
            }
        } else if (num instanceof BigDecimal) {
            BigDecimal bd = (BigDecimal) num;
            if (bd.compareTo(BigDecimal.ZERO) == 0 && bd.signum() < 0) {
                sb.append("-0");
            } else {
                sb.append(bd.stripTrailingZeros().toPlainString());
            }
        } else {
            sb.append(num);
        }
    }

    private static void printString(StringBuilder sb, CharSequence str) {
        sb.append('"');
        for (int i = 0; i < str.length(); i++) {
            char ch = str.charAt(i);
            switch (ch) {
                case '"':
                    sb.append("\\\"");
                    break;
                case '\\':
                    sb.append("\\\\");
                    break;
                case '/':
                    sb.append("\\/");
                    break;
                case '\b':
                    sb.append("\\b");
                    break;
                case '\f':
                    sb.append("\\f");
                    break;
                case '\n':
                    sb.append("\\n");
                    break;
                case '\r':
                    sb.append("\\r");
                    break;
                case '\t':
                    sb.append("\\t");
                    break;
                default:
                    if (ch >= 0x20 && ch != 0x7F) {
                        sb.append(ch);
                    } else {
                        appendUnicode(sb, ch);
                    }
                    break;
            }
        }
        sb.append('"');
    }

    private static void appendUnicode(StringBuilder sb, int cp) {
        sb.append("\\u");
        String hex = Integer.toHexString(cp);
        for (int i = hex.length(); i < 4; i++) {
            sb.append('0');
        }
        sb.append(hex);
    }

    private static void printArray(
            StringBuilder sb, List<?> list, boolean pretty, int indent, Set<Object> visiting) {
        enterContainer(list, visiting);
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

    private static void printObject(
            StringBuilder sb, Map<?, ?> map, boolean pretty, int indent, Set<Object> visiting) {
        enterContainer(map, visiting);
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
                sb.append(pretty ? " : " : ":");
                print(sb, entry.getValue(), pretty, indent + 1, visiting);
                first = false;
            }
            appendPrettyIndent(sb, pretty && !first, indent);
            sb.append('}');
        } finally {
            exitContainer(map, visiting);
        }
    }

    private static void enterContainer(Object container, Set<Object> visiting) {
        if (!visiting.add(container)) {
            throw new IllegalArgumentException("Cannot serialize cyclic structure");
        }
    }

    private static void exitContainer(Object container, Set<Object> visiting) {
        visiting.remove(container);
    }

    private static void appendPrettyIndent(StringBuilder sb, boolean enabled, int indent) {
        if (enabled) {
            sb.append('\n').append("  ".repeat(indent));
        }
    }

    // === ParseOptions ===

    public static final class ParseOptions {
        private boolean useBigDecimal = true;
        private int maxDepth = 1000;

        private ParseOptions() {}

        /** Create default parse options (BigDecimal for decimals, depth 1000). */
        public static ParseOptions defaults() {
            return new ParseOptions();
        }

        /** Create options configured for BigDecimal decimal parsing. */
        public static ParseOptions bigDecimal() {
            return defaults().useBigDecimalForFloats();
        }

        /** Create options configured for Double decimal parsing. */
        public static ParseOptions doublePrecision() {
            return defaults().useDoubleForFloats();
        }

        public ParseOptions useBigDecimalForFloats() {
            this.useBigDecimal = true;
            return this;
        }

        public ParseOptions useDoubleForFloats() {
            this.useBigDecimal = false;
            return this;
        }

        public ParseOptions maxParsingDepth(int depth) {
            if (depth <= 0) {
                throw new IllegalArgumentException("Maximum parsing depth must be positive");
            }
            this.maxDepth = depth;
            return this;
        }

        public boolean shouldUseBigDecimal() {
            return useBigDecimal;
        }

        public int getMaxParsingDepth() {
            return maxDepth;
        }
    }

    // === ParseException ===

    public static final class ParseException extends RuntimeException {
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

        public ParseException(String message, int line, int column) {
            super("Line " + line + ", Column " + column + ": " + message);
            this.position = -1;
            this.line = line;
            this.column = column;
        }

        public ParseException(String message, int line, int column, Throwable cause) {
            super("Line " + line + ", Column " + column + ": " + message, cause);
            this.position = -1;
            this.line = line;
            this.column = column;
        }

        public ParseException(String message, int position, int line, int column) {
            super(formatMessage(message, line, column, null, position));
            this.position = position;
            this.line = line;
            this.column = column;
        }

        public ParseException(String message, int position, int line, int column, Throwable cause) {
            super(formatMessage(message, line, column, null, position), cause);
            this.position = position;
            this.line = line;
            this.column = column;
        }

        public ParseException(
                String message, int position, int line, int column, CharSequence input) {
            super(formatMessage(message, line, column, input, position));
            this.position = position;
            this.line = line;
            this.column = column;
        }

        public ParseException(
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
}
