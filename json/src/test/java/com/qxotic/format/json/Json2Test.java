package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

class Json2Test {

    @Test
    void testNumberParsing() {
        assertEquals(0L, Json.parse("0"));
        assertEquals(123L, Json.parse("123"));
        assertEquals(-123L, Json.parse("-123"));
        assertEquals(3.14, ((Number) Json.parse("3.14")).doubleValue(), 0.0001);
        assertEquals(1.0, ((Number) Json.parse("1.0")).doubleValue(), 0.0001);
        assertEquals(0.0, ((Number) Json.parse("-0")).doubleValue(), 0.0001);
        assertEquals(10000000000.0, ((Number) Json.parse("1e10")).doubleValue(), 0.0001);
        assertEquals(0.01, ((Number) Json.parse("1E-2")).doubleValue(), 0.0001);

        Object big = Json.parse("9999999999999999999999999999999999999999");
        assertInstanceOf(BigInteger.class, big);
    }

    @Test
    void testStringParsing() {
        assertEquals("", Json.parse("\"\""));
        assertEquals("hello", Json.parse("\"hello\""));
        assertEquals("\"quoted\"", Json.parse("\"\\\"quoted\\\"\""));
        assertEquals("line1\nline2", Json.parse("\"line1\\nline2\""));
        assertEquals("A", Json.parse("\"\\u0041\""));
        assertEquals("\uD83D\uDE00", Json.parse("\"\\uD83D\\uDE00\""));
    }

    @Test
    void testEscapeSequences() {
        assertEquals("\\", Json.parse("\"\\\\\""));
        assertEquals("/", Json.parse("\"\\/\""));
        assertEquals("\b", Json.parse("\"\\b\""));
        assertEquals("\f", Json.parse("\"\\f\""));
        assertEquals("\n", Json.parse("\"\\n\""));
        assertEquals("\r", Json.parse("\"\\r\""));
        assertEquals("\t", Json.parse("\"\\t\""));
    }

    @Test
    void testArraysAndObjects() {
        List<Object> arr = new ArrayList<>();
        arr.add(1L);
        arr.add("two");
        arr.add(true);
        arr.add(Json.NULL);
        arr.add(Json.NULL);
        assertEquals(arr, Json.parse("[1,\"two\",true,null,null]"));

        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("name", "John");
        obj.put("age", 30L);
        obj.put("active", true);
        assertEquals(obj, Json.parse("{\"name\":\"John\",\"age\":30,\"active\":true}"));

        assertEquals(new ArrayList<>(), Json.parse("[]"));
        assertEquals(new LinkedHashMap<>(), Json.parse("{}"));
    }

    @Test
    void testNumberFormatting() {
        assertEquals("1", Json.stringify(1L));
        assertEquals("-1", Json.stringify(-1L));
        assertEquals("0", Json.stringify(0L));
        assertEquals("1.5", Json.stringify(1.5));
        assertEquals("1", Json.stringify(1.0));

        List<Object> list = new ArrayList<>();
        list.add(1L);
        list.add(2.5);
        list.add("hello");
        assertEquals("[1,2.5,\"hello\"]", Json.stringify(list));
    }

    @Test
    void testInvalidCases() {
        Json.ParseException e;

        e = assertThrows(Json.ParseException.class, () -> Json.parse("01"));
        assertTrue(e.getMessage().contains("Leading zeros"));

        assertThrows(Json.ParseException.class, () -> Json.parse("+1"));
        assertThrows(Json.ParseException.class, () -> Json.parse(".5"));
        assertThrows(Json.ParseException.class, () -> Json.parse("1."));
        assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        assertThrows(Json.ParseException.class, () -> Json.parse("\"line1\nline2\""));
        assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uD800\""));
        assertThrows(Json.ParseException.class, () -> Json.parse("\"\\uDC00\""));
    }
}
