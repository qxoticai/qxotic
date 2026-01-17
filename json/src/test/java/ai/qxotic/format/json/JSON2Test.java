package ai.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class JSON2Test {

    @Test
    void testNumberParsing() {
        assertEquals(0L, JSON.parse("0"));
        assertEquals(123L, JSON.parse("123"));
        assertEquals(-123L, JSON.parse("-123"));
        assertEquals(3.14, ((Number) JSON.parse("3.14")).doubleValue(), 0.0001);
        assertEquals(1.0, ((Number) JSON.parse("1.0")).doubleValue(), 0.0001);
        assertEquals(0.0, ((Number) JSON.parse("-0")).doubleValue(), 0.0001);
        assertEquals(10000000000.0, ((Number) JSON.parse("1e10")).doubleValue(), 0.0001);
        assertEquals(0.01, ((Number) JSON.parse("1E-2")).doubleValue(), 0.0001);

        Object big = JSON.parse("9999999999999999999999999999999999999999");
        assertTrue(big instanceof BigInteger);
    }

    @Test
    void testStringParsing() {
        assertEquals("", JSON.parse("\"\""));
        assertEquals("hello", JSON.parse("\"hello\""));
        assertEquals("\"quoted\"", JSON.parse("\"\\\"quoted\\\"\""));
        assertEquals("line1\nline2", JSON.parse("\"line1\\nline2\""));
        assertEquals("A", JSON.parse("\"\\u0041\""));
        assertEquals("\uD83D\uDE00", JSON.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testEscapeSequences() {
        assertEquals("\\", JSON.parse("\"\\\\\""));
        assertEquals("/", JSON.parse("\"\\/\""));
        assertEquals("\b", JSON.parse("\"\\b\""));
        assertEquals("\f", JSON.parse("\"\\f\""));
        assertEquals("\n", JSON.parse("\"\\n\""));
        assertEquals("\r", JSON.parse("\"\\r\""));
        assertEquals("\t", JSON.parse("\"\\t\""));
    }

    @Test
    void testArraysAndObjects() {
        List<Object> arr = new ArrayList<>();
        arr.add(1L);
        arr.add("two");
        arr.add(true);
        arr.add(JSON.NULL);
        arr.add(JSON.NULL);
        assertEquals(arr, JSON.parse("[1,\"two\",true,null,null]"));

        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("name", "John");
        obj.put("age", 30L);
        obj.put("active", true);
        assertEquals(obj, JSON.parse("{\"name\":\"John\",\"age\":30,\"active\":true}"));

        assertEquals(new ArrayList<>(), JSON.parse("[]"));
        assertEquals(new LinkedHashMap<>(), JSON.parse("{}"));
    }

    @Test
    void testNumberFormatting() {
        assertEquals("1", JSON.stringify(1L));
        assertEquals("-1", JSON.stringify(-1L));
        assertEquals("0", JSON.stringify(0L));
        assertEquals("1.5", JSON.stringify(1.5));
        assertEquals("1", JSON.stringify(1.0));

        List<Object> list = new ArrayList<>();
        list.add(1L);
        list.add(2.5);
        list.add("hello");
        assertEquals("[1,2.5,\"hello\"]", JSON.stringify(list));
    }

    @Test
    void testInvalidCases() {
        JSON.ParseException e;

        e = assertThrows(JSON.ParseException.class, () -> JSON.parse("01"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        assertThrows(JSON.ParseException.class, () -> JSON.parse("+1"));
        assertThrows(JSON.ParseException.class, () -> JSON.parse(".5"));
        assertThrows(JSON.ParseException.class, () -> JSON.parse("1."));
        assertThrows(JSON.ParseException.class, () -> JSON.parse("1e"));
        assertThrows(JSON.ParseException.class, () -> JSON.parse("\"line1\nline2\""));
        assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uD800\""));
        assertThrows(JSON.ParseException.class, () -> JSON.parse("\"\\uDC00\""));
    }
}
