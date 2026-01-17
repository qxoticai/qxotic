package ai.qxotic.format.json;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

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
        return parse(json, ParseOptions.create());
    }

    /** Parse JSON with custom options. */
    public static Object parse(CharSequence json, ParseOptions options) {
        Parser p = new Parser(json, options);
        Object result = p.parseValue();
        p.expectEnd();
        return result;
    }

    /** Convert Java object to compact JSON string. */
    public static String stringify(Object value) {
        return stringify(value, false);
    }

    /** Convert to JSON with optional pretty-printing. */
    public static String stringify(Object value, boolean pretty) {
        StringBuilder sb = new StringBuilder();
        print(sb, value, pretty, 0);
        return sb.toString();
    }

    // === Parser ===

    private static final class Parser {
        private final CharSequence input;
        private final ParseOptions options;
        private int pos;
        private int line = 1;
        private int col = 1;
        private int depth = 0;

        Parser(CharSequence input, ParseOptions options) {
            this.input = input;
            this.options = options;
        }

        Object parseValue() {
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
                case '-':
                case '0':
                case '1':
                case '2':
                case '3':
                case '4':
                case '5':
                case '6':
                case '7':
                case '8':
                case '9':
                    return parseNumber();
                case ']':
                    next();
                    throw error("Expected value");
                case '}':
                case ',':
                    next();
                    throw error("Unexpected character");
                default:
                    next();
                    throw error("Unexpected character");
            }
        }

        String parseString() {
            expect('"');
            StringBuilder sb = new StringBuilder();

            while (true) {
                char ch = next();
                if (ch == '"') break;
                if (ch == '\\') parseEscape(sb);
                else if (isValidUnescaped(ch)) sb.append(ch);
                else throw error("Control character must be escaped");
            }

            return sb.toString();
        }

        void parseEscape(StringBuilder sb) {
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
                    parseUnicode(sb);
                    break;
                default:
                    throw error("Invalid escape sequence");
            }
        }

        void parseUnicode(StringBuilder sb) {
            int code = parseHex(4);

            if (Character.isHighSurrogate((char) code)) {
                if (pos + 6 > input.length()
                        || input.charAt(pos) != '\\'
                        || input.charAt(pos + 1) != 'u') {
                    throw error("Lone surrogate");
                }
                pos += 2; // Skip backslash-u
                int low = parseHex(4);
                if (!Character.isLowSurrogate((char) low)) {
                    throw error("Unexpected character after high surrogate");
                }
                sb.append((char) code).append((char) low);
            } else if (Character.isLowSurrogate((char) code)) {
                throw error("Lone surrogate");
            } else {
                sb.appendCodePoint(code);
            }
        }

        int parseHex(int digits) {
            if (pos + digits > input.length()) {
                throw error("Incomplete Unicode escape");
            }

            int result = 0;
            for (int i = 0; i < digits; i++) {
                char ch = input.charAt(pos++);
                col++;
                int digit = Character.digit(ch, 16);
                if (digit < 0) throw error("Invalid hex digit");
                result = (result << 4) | digit;
            }
            return result;
        }

        List<Object> parseArray() {
            enterDepth();
            try {
                List<Object> list = new ArrayList<>();

                while (true) {
                    skipSpace();
                    char ch = peek();
                    if (ch == ']') break;
                    if (ch == '\0') throw error("Expected ']'");

                    if (!list.isEmpty()) {
                        skipSpace();
                        expect(',');
                    }

                    list.add(parseValue());
                }

                skipSpace();
                expect(']');
                return list;
            } finally {
                exitDepth();
            }
        }

        Map<String, Object> parseObject() {
            enterDepth();
            try {
                Map<String, Object> map = new LinkedHashMap<>();

                while (true) {
                    skipSpace();
                    char ch = peek();
                    if (ch == '}') break;
                    if (ch == '\0') throw error("Expected '}'");

                    if (!map.isEmpty()) {
                        skipSpace();
                        expect(',');
                    }

                    skipSpace();
                    String key = parseString();
                    skipSpace();
                    expect(':');
                    map.put(key, parseValue());
                }

                skipSpace();
                expect('}');
                return map;
            } finally {
                exitDepth();
            }
        }

        Number parseNumber() {
            int start = pos;

            // Sign
            if (peek() == '-') next();
            else if (peek() == '+') throw error("Unexpected '+'");

            // Integer part
            if (!isDigit(peek())) {
                if (peek() == '.') throw error("Unexpected '.'");
                throw error("Expected digit");
            }

            // No leading zeros (except single zero)
            if (peek() == '0' && pos + 1 < input.length() && isDigit(input.charAt(pos + 1))) {
                throw error("Leading zeros not allowed");
            }

            while (isDigit(peek())) next();

            // Fraction
            if (peek() == '.') {
                next();
                if (!isDigit(peek())) throw error("Expected digit after decimal point");
                while (isDigit(peek())) next();
            }

            // Exponent
            if (peek() == 'e' || peek() == 'E') {
                next();
                if (peek() == '+' || peek() == '-') next();
                if (!isDigit(peek())) throw error("exponent missing digits");
                while (isDigit(peek())) next();
            }

            String numStr = input.subSequence(start, pos).toString();
            boolean isDecimal =
                    numStr.contains(".") || numStr.contains("e") || numStr.contains("E");

            if (!isDecimal) {
                if (numStr.equals("-0")) return new BigDecimal("-0");
                try {
                    return Long.parseLong(numStr);
                } catch (NumberFormatException e) {
                    return new BigInteger(numStr);
                }
            } else {
                if (options.shouldUseBigDecimal()) {
                    try {
                        BigDecimal bd = new BigDecimal(numStr);
                        if (bd.compareTo(BigDecimal.ZERO) == 0) {
                            bd = bd.stripTrailingZeros();
                        }
                        return bd;
                    } catch (NumberFormatException e) {
                        throw error("Invalid number", e);
                    }
                } else {
                    try {
                        return Double.parseDouble(numStr);
                    } catch (NumberFormatException e) {
                        throw error("Invalid number", e);
                    }
                }
            }
        }

        // === Utilities ===

        private void skipSpace() {
            while (pos < input.length()) {
                char ch = input.charAt(pos);
                if (ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r') {
                    if (ch == '\n') {
                        line++;
                        col = 1;
                    } else col++;
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
            if (pos >= input.length()) throw error("Unexpected end of input");
            char ch = input.charAt(pos++);
            if (ch == '\n') {
                line++;
                col = 1;
            } else col++;
            return ch;
        }

        private void expect(char ch) {
            if (pos >= input.length()) throw error("Expected '" + ch + "'");
            if (next() != ch) throw error("Expected '" + ch + "'");
        }

        private void expect(String str) {
            for (int i = 0; i < str.length(); i++) {
                if (pos >= input.length()) throw error("Expected '" + str + "'");
                if (next() != str.charAt(i)) throw error("Expected '" + str + "'");
            }
        }

        private void expectEnd() {
            skipSpace();
            if (pos < input.length()) throw error("Expected end of input");
        }

        private void enterDepth() {
            if (++depth > options.getMaxParsingDepth())
                throw error("Maximum parsing depth exceeded");
        }

        private void exitDepth() {
            depth--;
        }

        private ParseException error(String msg) {
            int errorCol = col > 1 ? col - 1 : 1;
            return new ParseException(msg, line, errorCol);
        }

        private ParseException error(String msg, Throwable cause) {
            return new ParseException(msg, line, col, cause);
        }

        private static boolean isDigit(char ch) {
            return ch >= '0' && ch <= '9';
        }

        private static boolean isValidUnescaped(int cp) {
            return (cp >= 0x20 && cp <= 0x21)
                    || (cp >= 0x23 && cp <= 0x5B)
                    || (cp >= 0x5D && cp <= 0x10FFFF);
        }
    }

    // === Printer ===

    private static void print(StringBuilder sb, Object value, boolean pretty, int indent) {
        if (value == null || value == NULL) {
            sb.append("null");
        } else if (value instanceof Boolean) {
            sb.append(value);
        } else if (value instanceof Number) {
            printNumber(sb, (Number) value);
        } else if (value instanceof CharSequence) {
            printString(sb, (CharSequence) value);
        } else if (value instanceof List) {
            printArray(sb, (List<?>) value, pretty, indent);
        } else if (value instanceof Map) {
            printObject(sb, (Map<?, ?>) value, pretty, indent);
        } else {
            throw new IllegalArgumentException("Cannot serialize: " + value.getClass());
        }
    }

    private static void printNumber(StringBuilder sb, Number num) {
        if (num instanceof Float || num instanceof Double) {
            double d = num.doubleValue();
            if (Double.isNaN(d) || Double.isInfinite(d)) {
                throw new IllegalArgumentException("Cannot serialize NaN/Infinity");
            }
            if (d == (long) d) sb.append((long) d);
            else sb.append(d);
        } else if (num instanceof BigDecimal) {
            BigDecimal bd = (BigDecimal) num;
            if (bd.compareTo(BigDecimal.ZERO) == 0 && bd.signum() < 0) sb.append("-0");
            else sb.append(bd.stripTrailingZeros().toPlainString());
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
                    if (ch >= 0x20 && ch != 0x7F) sb.append(ch);
                    else appendUnicode(sb, ch);
                    break;
            }
        }
        sb.append('"');
    }

    private static void appendUnicode(StringBuilder sb, int cp) {
        sb.append("\\u");
        String hex = Integer.toHexString(cp);
        for (int i = hex.length(); i < 4; i++) sb.append('0');
        sb.append(hex);
    }

    private static void printArray(StringBuilder sb, List<?> list, boolean pretty, int indent) {
        sb.append('[');
        boolean first = true;
        for (Object elem : list) {
            if (!first) sb.append(',');
            if (pretty) sb.append('\n').append("  ".repeat(indent + 1));
            print(sb, elem, pretty, indent + 1);
            first = false;
        }
        if (pretty && !first) sb.append('\n').append("  ".repeat(indent));
        sb.append(']');
    }

    private static void printObject(StringBuilder sb, Map<?, ?> map, boolean pretty, int indent) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<?, ?> entry : map.entrySet()) {
            if (!first) sb.append(',');
            if (pretty) sb.append('\n').append("  ".repeat(indent + 1));
            printString(sb, (CharSequence) entry.getKey());
            sb.append(pretty ? " : " : ":");
            print(sb, entry.getValue(), pretty, indent + 1);
            first = false;
        }
        if (pretty && !first) sb.append('\n').append("  ".repeat(indent));
        sb.append('}');
    }

    // === ParseOptions ===

    public static final class ParseOptions {
        private boolean useBigDecimal = true;
        private int maxDepth = 1000;

        private ParseOptions() {}

        public static ParseOptions create() {
            return new ParseOptions();
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
            if (depth <= 0)
                throw new IllegalArgumentException("Maximum parsing depth must be positive");
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
        private final int line;
        private final int column;

        public ParseException(String message) {
            super(message);
            this.line = -1;
            this.column = -1;
        }

        public ParseException(String message, Throwable cause) {
            super(message, cause);
            this.line = -1;
            this.column = -1;
        }

        public ParseException(String message, int line, int column) {
            super("Line " + line + ", Column " + column + ": " + message);
            this.line = line;
            this.column = column;
        }

        public ParseException(String message, int line, int column, Throwable cause) {
            super("Line " + line + ", Column " + column + ": " + message, cause);
            this.line = line;
            this.column = column;
        }

        public int getLine() {
            return line;
        }

        public int getColumn() {
            return column;
        }
    }
}
