package com.llm4j.gguf.impl;

import java.lang.reflect.Array;

/**
 * Utility class for creating compact string representations of values.
 * Useful for logging, debugging, and display purposes where space is limited.
 */
class CompactFormatter {
    private static final int DEFAULT_MAX_LENGTH = 64;
    private static final int DEFAULT_LIST_ITEMS = 4;
    private static final String ELLIPSIS = "...";

    /**
     * Creates a compact string representation of an object.
     *
     * @param obj the object to format
     * @return a compact string representation
     */
    public static String format(Object obj) {
        return format(obj, DEFAULT_MAX_LENGTH, DEFAULT_LIST_ITEMS);
    }

    /**
     * Creates a compact string representation of an object with custom limits.
     *
     * @param obj       the object to format
     * @param maxLength maximum length for strings
     * @param maxItems  maximum number of items to show in collections
     * @return a compact string representation
     */
    public static String format(Object obj, int maxLength, int maxItems) {
        if (obj == null) {
            return "null";
        }

        // Handle arrays
        if (obj.getClass().isArray()) {
            return formatArray(obj, maxItems);
        }

        // Handle strings
        if (obj instanceof String) {
            return formatString((String) obj, maxLength, true);
        }

        // For all other types, use toString() and truncate if needed
        return formatString(obj.toString(), maxLength, false);
    }

    private static String formatString(String str, int maxLength, boolean addQuotes) {
        String value = formatString(str, maxLength);
        if (addQuotes) {
            value = "\"" + value + "\"";
        }
        return value;
    }

    private static String formatString(String str, int maxLength) {
        if (str.length() <= maxLength) {
            return str;
        }

        // Reserve space for ellipsis and quotes if present
        int targetLength = maxLength - ELLIPSIS.length();
        return str.substring(0, targetLength) + ELLIPSIS;
    }

    private static String formatArray(Object array, int maxItems) {
        int length = Array.getLength(array);
        if (length == 0) {
            return "[]";
        }

        StringBuilder sb = new StringBuilder("[");
        int itemsToShow = Math.min(length, maxItems);

        for (int i = 0; i < itemsToShow; i++) {
            if (i > 0) {
                sb.append(", ");
            }
            Object item = Array.get(array, i);
            sb.append(format(item, DEFAULT_MAX_LENGTH, maxItems));
        }

        if (length > maxItems) {
            sb.append(", ").append(ELLIPSIS);
        }

        return sb.append("]").toString();
    }
}
