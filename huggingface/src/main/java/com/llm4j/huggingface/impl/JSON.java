package com.llm4j.huggingface.impl;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;

/**
 * A lightweight JSON parser and printer utility class.
 * This class provides functionality to parse JSON strings into Java objects and
 * convert Java objects back to JSON strings.
 *
 * <p>The parser supports all standard JSON data types:
 * <ul>
 *   <li>Objects ({@code Map<String, Object>})</li>
 *   <li>Arrays ({@code List<Object>})</li>
 *   <li>Strings</li>
 *   <li>Numbers (both integer and floating-point)</li>
 *   <li>Booleans</li>
 *   <li>null</li>
 * </ul>
 *
 * <p>Usage example:
 * <pre>{@code
 * // Parsing JSON
 * String jsonStr = "{\"name\":\"John\",\"age\":30}";
 * Object parsed = JSON.parse(jsonStr);
 *
 * // Converting to JSON
 * var map = Map.of("key", List.of(JSON.NULL, 123, "foo", Math.PI)
 * String json = JSON.print(map, true);  // Pretty-printed
 * }</pre>
 */
public final class JSON {

    /**
     * Represents a JSON null value. Used to distinguish between Java null and JSON null.
     */
    public static final Object NULL = new Object() {
        @Override
        public String toString() {
            return "JSON.NULL";
        }
    };

    /**
     * Parses a JSON string into corresponding Java objects.
     *
     * @param chars The JSON string to parse
     * @return The parsed object, which can be one of:
     * {@code Map<String, Object>} for JSON objects,
     * {@code List<Object>} for JSON arrays,
     * {@code String} for JSON strings,
     * {@code Number} for JSON numbers,
     * {@code Boolean} for JSON booleans,
     * or {@code JSON.NULL} for JSON null
     * @throws ParseException if the input is not valid JSON
     */
    public static Object parse(CharSequence chars) {
        JSON json = new JSON(chars);
        Object result = json.parse();
        json.consumeEndOfInput(true);
        return result;
    }

    /**
     * Converts a Java object into a JSON string.
     *
     * @param json   The object to convert to JSON
     * @param pretty Whether to format the output with indentation and line breaks
     * @return A JSON string representation of the input object
     * @throws IllegalArgumentException if the input contains objects that cannot be converted to JSON
     */
    public static String print(Object json, boolean pretty) {
        StringBuilder sb = new StringBuilder();
        printImpl(sb, json, pretty, 0);
        return sb.toString();
    }

    // Private implementation details

    private final CharSequence chars;
    private int index;

    private JSON(CharSequence chars) {
        this.chars = chars;
        this.index = 0;
    }

    /**
     * Main parsing method that handles all JSON value types.
     */
    private Object parse() {
        char ch = currentChar(true);
        switch (ch) {
            case 't':
                consume("true");
                return true;
            case 'f':
                consume("false");
                return false;
            case 'n':
                consume("null");
                return JSON.NULL;
            case '"':
                return parseString();
            case '[':
                return parseList();
            case '{':
                return parseMap();
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
            default:
                throw parseException("Unexpected character " + getHexDigits(ch));
        }
    }

    /**
     * Parses a JSON array into a List.
     */
    private List<Object> parseList() {
        consume('[', true);
        List<Object> list = new ArrayList<>();
        while (currentChar(true) != ']') {
            list.add(parse());
            if (currentChar(true) == ',') {
                ++index;
            } else {
                break;
            }
        }
        consume(']', true);
        return list;
    }

    private Map<String, Object> parseMap() {
        consume('{', true);
        Map<String, Object> map = new LinkedHashMap<>();
        while (currentChar(true) != '}') {
            String key = parseString();
            consume(':', true);
            Object value = parse();
            map.put(key, value);
            if (currentChar(true) == ',') {
                ++index;
            } else {
                break;
            }
        }
        consume('}', true);
        return map;
    }

    private String parseString() {
        consume('"', true);
        StringBuilder sb = new StringBuilder();
        char ch;
        while ((ch = currentChar()) != '"') {
            if (ch == '\\') {
                index++;
                char escapeSuffix = currentChar();
                index++;
                switch (escapeSuffix) {
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
                        int result = 0;
                        for (int i = 0; i < 4; ++i) {
                            result = (result << 4) | Character.digit(parseHexDigit(), 16);
                        }
                        sb.append((char) result);
                        break;
                    default:
                        throw parseException("Unsupported escape sequence '\\" + escapeSuffix + "'");
                }
            } else {
                boolean isUnescaped = isUnescaped(ch);
                if (isUnescaped) {
                    sb.append(ch);
                    index++;
                } else {
                    throw parseException("Expected unescaped character but found '" + ch + "'");
                }
            }
        }
        consume('"');
        return sb.toString();
    }

    private static boolean isUnescaped(int codePoint) {
        return within(codePoint, 0x20, 0x21)
                || within(codePoint, 0x23, 0x5B)
                || within(codePoint, 0x5D, 0x10FFFF);
    }

    private Number parseNumber() {
        skipWhitespaces();
        int startIndex = index;
        while (index < chars.length() && "-+.eE0123456789".indexOf(currentChar()) >= 0) {
            ++index;
        }
        String numberStr = chars.subSequence(startIndex, index).toString();
        boolean isIntegral = numberStr.indexOf('.') < 0 && numberStr.indexOf('e') < 0 && numberStr.indexOf('E') < 0;
        if (isIntegral) {
            try {
                return Long.parseLong(numberStr);
            } catch (NumberFormatException e) {
                try {
                    return new BigInteger(numberStr);
                } catch (NumberFormatException e2) {
                    throw parseException("Cannot parse number", e2);
                }
            }
        } else {
            try {
                return Double.parseDouble(numberStr);
            } catch (NumberFormatException e) {
                try {
                    return new BigDecimal(numberStr);
                } catch (NumberFormatException e2) {
                    throw parseException("Cannot parse number", e);
                }
            }
        }
    }

    private char parseHexDigit() {
        char ch = currentChar();
        if (isDigit(ch) || within(ch, 'a', 'f') || within(ch, 'A', 'F')) {
            index++;
            return ch;
        } else {
            throw parseException("Expected hex digit 0-9a-fA-F but found '" + ch + "'");
        }
    }

    private ParseException parseException(String message, Throwable cause) {
        throw new ParseException(index + ": " + message, cause);
    }

    private ParseException parseException(String message) {
        throw new ParseException(index + ": " + message);
    }

    private void skipWhitespaces() {
        while (index < chars.length() && isWhitespace(chars.charAt(index))) {
            ++index;
        }
    }

    private static boolean isWhitespace(char ch) {
        return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r';
    }

    private void consume(char ch, boolean skipWhitespaces) {
        if (skipWhitespaces) {
            skipWhitespaces();
        }
        consume(ch);
    }

    private void consume(char ch) {
        if (currentChar() == ch) {
            ++index;
        } else {
            throw parseException("unexpected char");
        }
    }

    private void consume(String expected) {
        if (index > chars.length() - expected.length()) {
            throw parseException("expected \'" + expected + "\" but end-of-input found");
        }
        for (int i = 0; i < expected.length(); ++i) {
            if (expected.charAt(i) != chars.charAt(index)) {
                throw parseException("expected \'" + expected + "\"");
            }
            ++index;
        }
    }

    private void consumeEndOfInput(boolean skipWhitespaces) {
        if (skipWhitespaces) {
            skipWhitespaces();
        }
        if (index < chars.length()) {
            throw parseException("Expected end-of-input");
        }
    }

    private static boolean isDigit(char ch) {
        return within(ch, '0', '9');
    }

    private static boolean within(char ch, char lo, char hi) {
        return lo <= ch && ch <= hi;
    }

    private static boolean within(int cp, int lo, int hi) {
        return lo <= cp && cp <= hi;
    }

    private char currentChar() {
        if (index < chars.length()) {
            return chars.charAt(index);
        } else {
            throw parseException("end-of-input");
        }
    }

    private char currentChar(boolean skipWhitespaces) {
        if (skipWhitespaces) {
            skipWhitespaces();
        }
        return currentChar();
    }

    private static void printImpl(StringBuilder sb, Object json, boolean pretty, int indentLevel) {
        if (Objects.requireNonNull(json) instanceof Boolean) {
            Boolean bool = (Boolean) Objects.requireNonNull(json);
            sb.append(bool);
        } else if (json instanceof Float) {
            Float f = (Float) json;
            if (f.isNaN() || f.isInfinite()) {
                throw new IllegalArgumentException("Cannot serialize float value " + f);
            }
            sb.append(f);
        } else if (json instanceof Double) {
            Double d = (Double) json;
            if (d.isNaN() || d.isInfinite()) {
                throw new IllegalArgumentException("Cannot serialize double value " + d);
            }
            sb.append(d);
        } else if (json instanceof Number) {
            Number number = (Number) json;
            sb.append(number);
        } else if (json instanceof CharSequence) {
            CharSequence string = (CharSequence) json;
            printString(sb, string);
        } else if (json instanceof List<?>) {
            List<?> list = (List<?>) json;
            sb.append('[');
            boolean first = true;
            for (Object elem : list) {
                if (first) {
                    first = false;
                } else {
                    sb.append(',');
                }
                if (pretty) {
                    sb.append(System.lineSeparator());
                    sb.append("  ".repeat(indentLevel + 1));
                }
                printImpl(sb, elem, pretty, indentLevel + 1);
            }
            if (pretty && !first) {
                sb.append(System.lineSeparator());
                sb.append("  ".repeat(indentLevel));
            }
            sb.append(']');
        } else if (json instanceof Map<?, ?>) {
            Map<?, ?> map = (Map<?, ?>) json;
            sb.append('{');
            boolean first = true;
            for (Map.Entry<?, ?> entry : map.entrySet()) {
                if (first) {
                    first = false;
                } else {
                    sb.append(',');
                }
                if (pretty) {
                    sb.append(System.lineSeparator());
                    sb.append("  ".repeat(indentLevel + 1));
                }
                CharSequence key = (CharSequence) entry.getKey();
                printImpl(sb, key, pretty, indentLevel);
                sb.append(pretty ? " : " : ":");
                printImpl(sb, entry.getValue(), pretty, indentLevel + 1);
            }
            if (!first) {
                sb.append(System.lineSeparator());
                sb.append("  ".repeat(indentLevel));
            }
            sb.append('}');
        } else {
            if (json == NULL) {
                sb.append("null");
            } else {
                throw new IllegalArgumentException("unexpected element of type " + json.getClass());
            }
        }
    }

    /**
     * string = quotation-mark *char quotation-mark
     * <p>
     * char = unescaped /
     * escape (
     * %x22 /          ; "    quotation mark  U+0022
     * %x5C /          ; \    reverse solidus U+005C
     * %x2F /          ; /    solidus         U+002F
     * %x62 /          ; b    backspace       U+0008
     * %x66 /          ; f    form feed       U+000C
     * %x6E /          ; n    line feed       U+000A
     * %x72 /          ; r    carriage return U+000D
     * %x74 /          ; t    tab             U+0009
     * %x75 4HEXDIG )  ; uXXXX                U+XXXX
     * <p>
     * escape = %x5C              ; \
     * <p>
     * quotation-mark = %x22      ; "
     * <p>
     * unescaped = %x20-21 / %x23-5B / %x5D-10FFFF
     */
    private static void printString(StringBuilder sb, CharSequence string) {
        sb.append('"');
        string.codePoints().forEachOrdered(cp -> {
            switch (cp) {
                case '"':
                    sb.append('\\').append('"');
                    break;
                case '\\':
                    sb.append('\\').append('\\');
                    break;
                case '/':
                    sb.append('\\').append('/');
                    break;
                case '\b':
                    sb.append('\\').append('b');
                    break;
                case '\f':
                    sb.append('\\').append('f');
                    break;
                case '\n':
                    sb.append('\\').append('n');
                    break;
                case '\r':
                    sb.append('\\').append('r');
                    break;
                case '\t':
                    sb.append('\\').append('t');
                    break;
                default:
                    if (isUnescaped(cp)) {
                        sb.appendCodePoint(cp);
                    } else {
                        if (Character.charCount(cp) == 1) {
                            sb.append("\\u").append(getHexDigits(cp));
                        } else {
                            sb.append("\\u").append(getHexDigits(Character.highSurrogate(cp)));
                            sb.append("\\u").append(getHexDigits(Character.lowSurrogate(cp)));
                        }
                    }
                    break;
            }
        });
        sb.append('"');
    }

    private static String getHexDigits(int cp) {
        String hexString = Integer.toHexString(cp);
        return "0".repeat(4 - hexString.length()) + hexString;
    }

    /**
     * Custom exception for JSON parsing errors.
     */
    public static final class ParseException extends RuntimeException {
        /**
         * Creates a new ParseException with the specified message.
         *
         * @param message The error message
         */
        public ParseException(String message) {
            super(message);
        }

        /**
         * Creates a new ParseException with the specified message and cause.
         *
         * @param message The error message
         * @param cause   The cause of the error
         */
        public ParseException(String message, Throwable cause) {
            super(message, cause);
        }
    }
}