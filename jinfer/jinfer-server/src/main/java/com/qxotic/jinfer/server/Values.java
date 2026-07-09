// Typed value coercion for parsed-JSON request maps: read objects/arrays/strings/numbers with
// defaults and lenient (string-encoded) number parsing. Shared leaf utility, no dependencies.
package com.qxotic.jinfer.server;

import com.qxotic.jinfer.*;
import java.util.List;
import java.util.Map;

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
        return value instanceof Boolean b ? b : defaultValue;
    }

    static int intValue(Object value, int defaultValue) {
        return Math.toIntExact(longValue(value, defaultValue));
    }

    static long longValue(Object value, long defaultValue) {
        if (value instanceof Number n) {
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
        return defaultValue;
    }

    /**
     * The text of an OpenAI message {@code content} field: a plain string, or the concatenated
     * {@code text} parts of a multimodal content array (non-text parts ignored).
     */
    static String messageContent(Object content) {
        if (content instanceof List<?> parts) {
            StringBuilder sb = new StringBuilder();
            for (Object part : parts) {
                if (part instanceof Map<?, ?> map && "text".equals(map.get("type"))) {
                    Object text = map.get("text");
                    if (text != null) sb.append(text);
                }
            }
            return sb.toString();
        }
        return stringValue(content, "");
    }

    static float floatValue(Object value, float defaultValue) {
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
        return defaultValue;
    }
}
