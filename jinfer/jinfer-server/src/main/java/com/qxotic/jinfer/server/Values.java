// Typed value coercion for parsed-JSON request maps: read objects/arrays/strings/numbers with
// defaults and lenient (string-encoded) number parsing. Shared leaf utility, no dependencies.
package com.qxotic.jinfer.server;

import java.util.List;
import java.util.Map;

/**
 * Loose-JSON coercions for request maps: each accessor states the expected shape and throws with
 * the offending KEY on mismatch - endpoint errors name the field, not a cast site.
 */
final class Values {
    private Values() {}

    @SuppressWarnings("unchecked")
    static Map<String, Object> asObject(Object value, String name) {
        if (value instanceof Map<?, ?> map) return (Map<String, Object>) map;
        throw new IllegalArgumentException(name + " must be an object");
    }

    @SuppressWarnings("unchecked")
    static List<Object> asArray(Object value, String name) {
        if (value instanceof List<?> list) return (List<Object>) list;
        throw new IllegalArgumentException(name + " must be an array");
    }

    static String stringValue(Object value, String defaultValue) {
        return value == null ? defaultValue : String.valueOf(value);
    }

    static boolean booleanValue(Object value, boolean defaultValue) {
        if (value == null) return defaultValue;
        if (value instanceof Boolean b) return b;
        throw new IllegalArgumentException("Invalid argument: '" + value + "' is not a boolean");
    }

    static int intValue(Object value, int defaultValue) {
        long wide = longValue(value, defaultValue);
        // toIntExact throws ArithmeticException, which is NOT one of the two types a server maps
        // to 400 - so an out-of-range number in a request came back as "Internal server error",
        // blaming the server for the client's 99999999999
        if (wide < Integer.MIN_VALUE || wide > Integer.MAX_VALUE) {
            throw new IllegalArgumentException(
                    "Invalid argument: " + wide + " is out of range for a 32-bit integer");
        }
        return (int) wide;
    }

    static long longValue(Object value, long defaultValue) {
        if (value == null) return defaultValue;
        if (value instanceof Number n) {
            if (n instanceof Byte
                    || n instanceof Short
                    || n instanceof Integer
                    || n instanceof Long) return n.longValue();
            double wide = n.doubleValue();
            if (!Double.isFinite(wide)
                    || wide != Math.rint(wide)
                    || wide < -0x1p63
                    || wide >= 0x1p63) {
                throw new IllegalArgumentException(
                        "Invalid argument: '" + n + "' is not an integer");
            }
            return n.longValue();
        }
        if (value instanceof String s) { // tolerate string-encoded numbers (e.g. "seed": "42")
            try {
                return Long.parseLong(s.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException(
                        "Invalid argument: '" + s + "' is not an integer");
            }
        }
        throw new IllegalArgumentException("Invalid argument: '" + value + "' is not an integer");
    }

    /**
     * The text of an OpenAI message {@code content} field: a plain string, or the concatenated
     * {@code text} parts of a multimodal content array (non-text parts ignored).
     */
    static String messageContent(Object content) {
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object part : parts) {
                if (part instanceof Map<?, ?> map
                        && List.of("text", "input_text", "output_text").contains(map.get("type"))) {
                    Object text = map.get("text") != null ? map.get("text") : map.get("input_text");
                    if (text != null) sb.append(text);
                }
            }
            return sb.toString();
        }
        return stringValue(content, "");
    }

    static float floatValue(Object value, float defaultValue) {
        if (value == null) return defaultValue;
        if (value instanceof Number n) {
            return n.floatValue();
        }
        if (value instanceof String s) {
            try {
                return Float.parseFloat(s.trim());
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Invalid argument: '" + s + "' is not a number");
            }
        }
        throw new IllegalArgumentException("Invalid argument: '" + value + "' is not a number");
    }
}
