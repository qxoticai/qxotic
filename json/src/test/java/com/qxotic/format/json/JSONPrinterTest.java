package com.qxotic.format.json;

import static com.qxotic.format.json.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;

class JSONPrinterTest {

    @Test
    void testCompactOutput() {
        assertEquals("{\"key\":\"value\"}", JSON.stringify(map("key", "value"), false));
        assertEquals("[1,2,3]", JSON.stringify(list(1, 2, 3), false));
    }

    @Test
    void testPrettyOutput() {
        String json = JSON.stringify(map("name", "John", "age", 30), true);
        assertTrue(json.contains("\"name\" : \"John\""));
        assertTrue(json.contains("\n"));
    }

    @Test
    void testNegativeZero() {
        assertEquals("0", JSON.stringify(new BigDecimal("-0"), false));
        assertEquals("{\"zero\":0}", JSON.stringify(map("zero", new BigDecimal("-0")), false));
    }

    @ParameterizedTest
    @MethodSource("numberPrintCases")
    void testNumberPrinting(Number input, String expected) {
        assertEquals(expected, JSON.stringify(input, false));
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
        assertEquals("true", JSON.stringify(true, false));
        assertEquals("false", JSON.stringify(false, false));
        assertEquals("null", JSON.stringify(JSON.NULL, false));
    }

    @Test
    void testStringEscapes() {
        assertEquals("\"hello\"", JSON.stringify("hello", false));
        assertEquals("\"\\\"test\\\"\"", JSON.stringify("\"test\"", false));
        assertEquals("\"\\\\test\"", JSON.stringify("\\test", false));
        assertEquals("\"line1\\nline2\"", JSON.stringify("line1\nline2", false));
        assertEquals("\"a\\tb\"", JSON.stringify("a\tb", false));
        assertEquals("\"中文\"", JSON.stringify("中文", false));
        assertEquals("\"\uD83D\uDE00\"", JSON.stringify("\uD83D\uDE00", false));
    }

    @Test
    void testEmptyStructures() {
        assertEquals("[]", JSON.stringify(list(), false));
        assertEquals("{}", JSON.stringify(map(), false));
        assertEquals("\"\"", JSON.stringify("", false));
    }

    @Test
    void testComplexStructure() {
        Map<String, Object> obj = map("array", list(1, 2, 3), "nested", map("a", 1));
        String json = JSON.stringify(obj, false);
        assertTrue(json.contains("\"array\":[1,2,3]"));
        assertTrue(json.contains("\"nested\":{\"a\":1}"));
    }

    @ParameterizedTest
    @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY})
    void testInvalidNumbersThrow(double value) {
        assertThrows(IllegalArgumentException.class, () -> JSON.stringify(value, false));
    }

    @Test
    void testCyclicStructuresThrow() {
        List<Object> cyclicList = new ArrayList<>();
        cyclicList.add("x");
        cyclicList.add(cyclicList);
        assertThrows(IllegalArgumentException.class, () -> JSON.stringify(cyclicList, false));

        Map<String, Object> cyclicMap = new LinkedHashMap<>();
        cyclicMap.put("self", cyclicMap);
        assertThrows(IllegalArgumentException.class, () -> JSON.stringify(cyclicMap, false));
    }

    @Test
    void testSharedReferencesAllowed() {
        List<Object> child = new ArrayList<>();
        child.add(1L);
        List<Object> root = new ArrayList<>();
        root.add(child);
        root.add(child);
        assertEquals("[[1],[1]]", JSON.stringify(root, false));
    }
}
