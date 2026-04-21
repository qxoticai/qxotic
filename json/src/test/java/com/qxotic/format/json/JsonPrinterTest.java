package com.qxotic.format.json;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.junit.jupiter.params.provider.ValueSource;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static com.qxotic.format.json.TestUtils.list;
import static com.qxotic.format.json.TestUtils.map;
import static org.junit.jupiter.api.Assertions.*;

class JsonPrinterTest {

    @Test
    void testCompactOutput() {
        assertEquals("{\"key\":\"value\"}", Json.stringify(map("key", "value"), false));
        assertEquals("[1,2,3]", Json.stringify(list(1, 2, 3), false));
    }

    @Test
    void testPrettyOutput() {
        String json = Json.stringify(map("name", "John", "age", 30), true);
        assertTrue(json.contains("\"name\": \"John\""));
        assertTrue(json.contains("\n"));
    }

    @Test
    void testNegativeZero() {
        assertEquals("0", Json.stringify(new BigDecimal("-0"), false));
        assertEquals("{\"zero\":0}", Json.stringify(map("zero", new BigDecimal("-0")), false));
    }

    @ParameterizedTest
    @MethodSource("numberPrintCases")
    void testNumberPrinting(Number input, String expected) {
        assertEquals(expected, Json.stringify(input, false));
    }

    static Stream<Arguments> numberPrintCases() {
        return Stream.of(
                Arguments.of(123L, "123"),
                Arguments.of(-456L, "-456"),
                Arguments.of(
                        new BigInteger("9999999999999999999999999999999999999999"),
                        "9999999999999999999999999999999999999999"),
                Arguments.of(new BigDecimal("123.456"), "123.456"),
                Arguments.of(new BigDecimal("1.5000"), "1.5"),
                Arguments.of(1.5, "1.5"),
                Arguments.of(2.0, "2"),
                Arguments.of(1.5f, "1.5"),
                Arguments.of(2.0f, "2"),
                Arguments.of(Float.valueOf(1.25f), "1.25"),
                Arguments.of(Double.valueOf(2.5d), "2.5"),
                Arguments.of(Integer.valueOf(7), "7"),
                Arguments.of(Long.valueOf(9L), "9"));
    }

    @Test
    void testLiteralPrinting() {
        assertEquals("true", Json.stringify(true, false));
        assertEquals("false", Json.stringify(false, false));
        assertEquals("null", Json.stringify(Json.NULL, false));
    }

    @Test
    void testStringEscapes() {
        assertEquals("\"hello\"", Json.stringify("hello", false));
        assertEquals("\"\\\"test\\\"\"", Json.stringify("\"test\"", false));
        assertEquals("\"\\\\test\"", Json.stringify("\\test", false));
        assertEquals("\"line1\\nline2\"", Json.stringify("line1\nline2", false));
        assertEquals("\"a\\tb\"", Json.stringify("a\tb", false));
        assertEquals("\"中文\"", Json.stringify("中文", false));
        assertEquals("\"\uD83D\uDE00\"", Json.stringify("\uD83D\uDE00", false));
    }

    @Test
    void testEmptyStructures() {
        assertEquals("[]", Json.stringify(list(), false));
        assertEquals("{}", Json.stringify(map(), false));
        assertEquals("\"\"", Json.stringify("", false));
    }

    @Test
    void testComplexStructure() {
        Map<String, Object> obj = map("array", list(1, 2, 3), "nested", map("a", 1));
        String json = Json.stringify(obj, false);
        assertTrue(json.contains("\"array\":[1,2,3]"));
        assertTrue(json.contains("\"nested\":{\"a\":1}"));
    }

    @ParameterizedTest
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void testInvalidNumbersThrow(double value) {
        assertThrows(IllegalArgumentException.class, () -> Json.stringify(value, false));
    }

    @Test
    void testCyclicStructuresThrow() {
        List<Object> cyclicList = new ArrayList<>();
        cyclicList.add("x");
        cyclicList.add(cyclicList);
        assertThrows(IllegalArgumentException.class, () -> Json.stringify(cyclicList, false));

        Map<String, Object> cyclicMap = new LinkedHashMap<>();
        cyclicMap.put("self", cyclicMap);
        assertThrows(IllegalArgumentException.class, () -> Json.stringify(cyclicMap, false));
    }

    @Test
    void testSharedReferencesAllowed() {
        List<Object> child = new ArrayList<>();
        child.add(1L);
        List<Object> root = new ArrayList<>();
        root.add(child);
        root.add(child);
        assertEquals("[[1],[1]]", Json.stringify(root, false));
    }
}
