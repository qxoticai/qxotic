package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/** Shared test utilities for JSON tests. */
public final class TestUtils {

    private TestUtils() {}

    /** Creates a LinkedHashMap with insertion order from key-value pairs. */
    public static Map<String, Object> map(Object... keyValues) {
        if (keyValues.length % 2 != 0) {
            throw new IllegalArgumentException("keyValues must have even length");
        }
        Map<String, Object> map = new LinkedHashMap<>();
        for (int i = 0; i < keyValues.length; i += 2) {
            map.put((String) keyValues[i], keyValues[i + 1]);
        }
        return map;
    }

    /** Creates a List from values. */
    @SafeVarargs
    public static List<Object> list(Object... values) {
        return new ArrayList<>(Arrays.asList(values));
    }

    /** Asserts that value round-trips correctly through stringify/parse. */
    public static void assertRoundTrip(Object value) {
        assertRoundTrip(value, false);
    }

    /** Asserts that value round-trips correctly, optionally pretty-printed. */
    public static void assertRoundTrip(Object value, boolean pretty) {
        String json = JSON.stringify(value, pretty);
        Object parsed = JSON.parse(json);
        assertTrue(deepEquals(value, parsed),
            "Round-trip failed for: " + value + " -> " + json + " -> " + parsed);
    }

    /** Asserts that parsing throws with expected message substring. */
    public static void assertParseThrows(String json, String expectedMessage) {
        JSON.ParseException e = assertThrows(JSON.ParseException.class, () -> JSON.parse(json));
        assertTrue(e.getMessage().contains(expectedMessage),
            "Expected message containing '" + expectedMessage + "' but got: " + e.getMessage());
    }

    /** Deep equals for JSON values. */
    public static boolean deepEquals(Object a, Object b) {
        if (a == b) return true;
        if (a == null || b == null) return false;
        if (a instanceof Number && b instanceof Number) {
            return ((Number) a).doubleValue() == ((Number) b).doubleValue();
        }
        if (a instanceof List && b instanceof List) {
            List<?> la = (List<?>) a;
            List<?> lb = (List<?>) b;
            if (la.size() != lb.size()) return false;
            for (int i = 0; i < la.size(); i++) {
                if (!deepEquals(la.get(i), lb.get(i))) return false;
            }
            return true;
        }
        if (a instanceof Map && b instanceof Map) {
            Map<?, ?> ma = (Map<?, ?>) a;
            Map<?, ?> mb = (Map<?, ?>) b;
            if (ma.size() != mb.size()) return false;
            for (Map.Entry<?, ?> e : ma.entrySet()) {
                if (!deepEquals(e.getValue(), mb.get(e.getKey()))) return false;
            }
            return true;
        }
        return a.equals(b);
    }
}
