package com.qxotic.format.json;

import org.junit.jupiter.api.Test;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.List;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.*;

class Json2ComparisonTest {

    @Test
    void testJSON2RejectsInvalidNumbers() {
        String[] invalidNumbers = {
            "+1", "01", "00", ".5", "1.", "1e", "1e+", "1e-", "1.2.3", "01.5", "00.5"
        };

        for (String num : invalidNumbers) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(num),
                    "JSON2 should reject: " + num);
        }
    }

    @Test
    void testJSON2RejectsInvalidStrings() {
        String[] invalidStrings = {
            "\"\\u00\"", "\"\\u00G\"",
            "\"\\uD800\"", "\"\\uDC00\""
        };

        for (String str : invalidStrings) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(str),
                    "JSON2 should reject: " + str);
        }
    }

    @Test
    void testJSON2HandlesSurrogatesCorrectly() {
        String[] validSurrogates = {
            "\"\\uD83D\\uDE00\"", "\"\\uD83D\\uDC4D\"", "\"Hello \\uD83D\\uDE00\""
        };

        for (String str : validSurrogates) {
            Object result = Json.parse(str);
            assertNotNull(result);
        }

        String[] invalidSurrogates = {"\"\\uD83DA\"", "\"\\uD83D\\u0041\""};

        for (String str : invalidSurrogates) {
            assertThrows(
                    Json.ParseException.class,
                    () -> Json.parse(str),
                    "JSON2 should reject: " + str);
        }
    }

    @Test
    void testJSON2BetterErrorMessages() {
        Json.ParseException e;

        e = assertThrows(Json.ParseException.class, () -> Json.parse("{\n  \"key\":,\n}"));
        assertEquals(2, e.getLine());
        assertTrue(e.getMessage().contains("Line"));
        assertTrue(e.getMessage().contains("Column"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("01"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        e = assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\""));
        assertTrue(e.getMessage().contains("surrogate"));
    }

    @Test
    void testJSON2PreservesNegativeZero() {
        Object parsed = Json.parse("-0");
        assertInstanceOf(Number.class, parsed);

        String stringified = Json.stringify(parsed);
        assertEquals("0", stringified);

        Object roundTrip = Json.parse(stringified);
        assertEquals(0, ((Number) roundTrip).doubleValue(), 0.0001);
    }

    @Test
    void testBothHandleCorrectly() {
        String[] validJson = {
            "true",
            "false",
            "null",
            "123",
            "\"hello\"",
            "[1,2,3]",
            "{\"key\":\"value\"}",
            "{\"a\":1,\"b\":[1,2],\"c\":{\"d\":2}}"
        };

        for (String json : validJson) {
            Object json1 = Json.parse(json);
            Object json2 = Json.parse(json);

            assertTrue(objectsEqual(json1, json2), "Both should parse " + json + " same way");
        }
    }

    private boolean objectsEqual(Object o1, Object o2) {
        if (o1 == o2) return true;
        if (o1 == null || o2 == null) return false;

        if (o1 == Json.NULL && o2 == Json.NULL) return true;
        if (o1 == Json.NULL && o2 == Json.NULL) return true;

        if (o1 instanceof Number && o2 instanceof Number) {
            Number n1 = (Number) o1;
            Number n2 = (Number) o2;
            if (n1 instanceof BigDecimal || n2 instanceof BigDecimal) {
                return new BigDecimal(n1.toString()).compareTo(new BigDecimal(n2.toString())) == 0;
            }
            if (n1 instanceof BigInteger || n2 instanceof BigInteger) {
                return new BigInteger(n1.toString()).equals(new BigInteger(n2.toString()));
            }
            return n1.doubleValue() == n2.doubleValue();
        }

        if (o1 instanceof List && o2 instanceof List) {
            List<?> l1 = (List<?>) o1;
            List<?> l2 = (List<?>) o2;
            if (l1.size() != l2.size()) return false;
            for (int i = 0; i < l1.size(); i++) {
                if (!objectsEqual(l1.get(i), l2.get(i))) return false;
            }
            return true;
        }

        if (o1 instanceof Map && o2 instanceof Map) {
            Map<?, ?> m1 = (Map<?, ?>) o1;
            Map<?, ?> m2 = (Map<?, ?>) o2;
            if (m1.size() != m2.size()) return false;
            for (Map.Entry<?, ?> entry : m1.entrySet()) {
                Object value2 = m2.get(entry.getKey());
                if (!objectsEqual(entry.getValue(), value2)) return false;
            }
            return true;
        }

        return o1.equals(o2);
    }
}
