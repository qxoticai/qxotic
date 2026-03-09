package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.*;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive edge case tests for JSON parsing and stringify. Tests corner cases not covered in
 * other test files.
 */
class JsonEdgeCaseTests {

    // ===== Number Edge Cases =====

    @Test
    void testNumberMaxLong() {
        assertEquals(Long.MAX_VALUE, Json.parse(String.valueOf(Long.MAX_VALUE)));
        assertEquals(String.valueOf(Long.MAX_VALUE), Json.stringify(Long.MAX_VALUE));
    }

    @Test
    void testNumberMinLong() {
        assertEquals(Long.MIN_VALUE, Json.parse(String.valueOf(Long.MIN_VALUE)));
        assertEquals(String.valueOf(Long.MIN_VALUE), Json.stringify(Long.MIN_VALUE));
    }

    @Test
    void testNumberAtLongBoundary() {
        // Just above Long.MAX_VALUE should be BigInteger
        String aboveMax = "9223372036854775808"; // Long.MAX_VALUE + 1
        Object parsed = Json.parse(aboveMax);
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger(aboveMax), parsed);

        // Just below Long.MIN_VALUE should be BigInteger
        String belowMin = "-9223372036854775809"; // Long.MIN_VALUE - 1
        parsed = Json.parse(belowMin);
        assertInstanceOf(BigInteger.class, parsed);
        assertEquals(new BigInteger(belowMin), parsed);
    }

    @Test
    void testNumberScientificNotationBounds() {
        // Very large exponent
        assertEquals(new BigDecimal("1E+308"), Json.parse("1e308"));
        assertEquals(new BigDecimal("1E-307"), Json.parse("1e-307"));

        // Test with BigDecimal mode
        Object parsed =
                Json.parse("1e308", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(0, new BigDecimal("1E+308").compareTo((BigDecimal) parsed));
    }

    @Test
    void testNumberExponentWithPlusSign() {
        assertEquals(new BigDecimal("1.5E+10"), Json.parse("1.5e+10"));
        assertEquals(new BigDecimal("1.5E-10"), Json.parse("1.5e-10"));
    }

    @Test
    void testNumberExponentUpperCase() {
        assertEquals(new BigDecimal("1.5E+10"), Json.parse("1.5E+10"));
        assertEquals(new BigDecimal("1.5E-10"), Json.parse("1.5E-10"));
    }

    @Test
    void testNumberTrailingDecimalZerosPreservedInBigDecimalMode() {
        // In BigDecimal mode, trailing zeros should be preserved except for zero values
        Object parsed =
                Json.parse("1.500", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("1.500"), parsed);

        parsed = Json.parse("1.500e2", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("150.0"), parsed);
    }

    @Test
    void testNumberVeryPreciseDecimal() {
        String precise = "0.1234567890123456789012345678901234567890";
        Object parsed =
                Json.parse(precise, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal(precise), parsed);
    }

    // ===== String Edge Cases =====

    @Test
    void testStringWithAllControlCharactersEscaped() {
        // Test all control characters that must be escaped
        for (int i = 0; i <= 0x1F; i++) {
            char c = (char) i;
            String input = "\"" + c + "\"";
            Json.ParseException e =
                    assertThrows(Json.ParseException.class, () -> Json.parse(input));
            assertTrue(e.getMessage().contains("Control") || e.getMessage().contains("control"));
        }
    }

    @Test
    void testStringWithValidEscapes() {
        assertEquals("\"", Json.parse("\"\\\"\""));
        assertEquals("\\", Json.parse("\"\\\\\""));
        assertEquals("/", Json.parse("\"\\/\""));
        assertEquals("\b", Json.parse("\"\\b\""));
        assertEquals("\f", Json.parse("\"\\f\""));
        assertEquals("\n", Json.parse("\"\\n\""));
        assertEquals("\r", Json.parse("\"\\r\""));
        assertEquals("\t", Json.parse("\"\\t\""));
    }

    @Test
    void testStringUnicodeSupplementaryPlanes() {
        // Test characters from supplementary planes
        String supplementary = "\uD83D\uDE00\uD83C\uDF89\uD83C\uDDFA\uD83C\uDDF8";
        String json = Json.stringify(supplementary, false);
        assertEquals("\"" + supplementary + "\"", json);

        // Round trip
        Object parsed = Json.parse(json);
        assertEquals(supplementary, parsed);
    }

    @Test
    void testStringMixedEscapedAndUnescapedUnicode() {
        String input = "A\u0042C\u0044E"; // A B C D E
        String json = Json.stringify(input, false);
        // Should output direct Unicode, not escaped
        assertEquals("\"ABCDE\"", json);
    }

    // ===== Array Edge Cases =====

    @Test
    void testEmptyArray() {
        assertEquals(Collections.emptyList(), Json.parse("[]"));
        assertEquals("[]", Json.stringify(Collections.emptyList(), false));
    }

    @Test
    void testSingleElementArray() {
        assertEquals(List.of(1L), Json.parse("[1]"));
        assertEquals(List.of("test"), Json.parse("[\"test\"]"));
        assertEquals(List.of(true), Json.parse("[true]"));
        assertEquals(List.of(Json.NULL), Json.parse("[null]"));
    }

    @Test
    void testArrayWithMixedTypes() {
        // Default mode uses BigDecimal for floating-point numbers
        List<Object> expected = Arrays.asList(1L, "two", true, Json.NULL, new BigDecimal("3.14"));
        Object parsed = Json.parse("[1, \"two\", true, null, 3.14]");
        // Compare content, not exact List implementation
        assertInstanceOf(List.class, parsed);
        List<?> parsedList = (List<?>) parsed;
        assertEquals(expected.size(), parsedList.size());
        for (int i = 0; i < expected.size(); i++) {
            if (expected.get(i) instanceof BigDecimal) {
                // Compare BigDecimal values
                assertEquals(
                        0,
                        ((BigDecimal) expected.get(i)).compareTo((BigDecimal) parsedList.get(i)));
            } else {
                assertEquals(expected.get(i), parsedList.get(i));
            }
        }
    }

    @Test
    void testArrayNestedEmpty() {
        assertEquals(List.of(Collections.emptyList()), Json.parse("[[]]"));
        assertEquals(List.of(Collections.emptyMap()), Json.parse("[{}]"));
    }

    // ===== Object Edge Cases =====

    @Test
    void testEmptyObject() {
        assertEquals(Collections.emptyMap(), Json.parse("{}"));
        assertEquals("{}", Json.stringify(Collections.emptyMap(), false));
    }

    @Test
    void testObjectWithNumericKeys() {
        // JSON keys must be strings, numbers as strings are valid
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("123", "value");
        expected.put("456", "another");

        Object parsed = Json.parse("{\"123\": \"value\", \"456\": \"another\"}");
        assertEquals(expected, parsed);
    }

    @Test
    void testObjectWithSpecialCharacterKeys() {
        Map<String, Object> expected = new LinkedHashMap<>();
        expected.put("key-with-dash", "value");
        expected.put("key.with.dot", "value");
        expected.put("key:with:colon", "value");

        Object parsed =
                Json.parse(
                        "{\"key-with-dash\": \"value\", \"key.with.dot\": \"value\","
                                + " \"key:with:colon\": \"value\"}");
        assertEquals(expected, parsed);
    }

    @Test
    void testObjectDuplicateKeysLastWins() {
        // RFC 8259 says duplicate names are allowed, last value wins
        Map<String, Object> parsed =
                (Map<String, Object>) Json.parse("{\"key\": \"first\", \"key\": \"last\"}");
        assertEquals(1, parsed.size());
        assertEquals("last", parsed.get("key"));
    }

    // ===== Depth and Recursion Edge Cases =====

    @Test
    void testMaxDepthExactlyAtLimit() {
        // Test parsing at exactly the maximum depth limit
        // Default max depth is 1000, but we use 200 to avoid Java recursion limits
        // and test with custom options for lower limits

        // Test with depth 200 (should parse with default max depth 1000)
        int depth = 200;
        StringBuilder sb = new StringBuilder();
        sb.append("[".repeat(depth));
        sb.append("null");
        sb.append("]".repeat(depth));

        Object result = Json.parse(sb.toString());
        assertNotNull(result);

        // Should fail with custom max depth of 100 (depth 200 > 100)
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () ->
                                Json.parse(
                                        sb.toString(), Json.ParseOptions.defaults().maxDepth(100)));
        assertNotNull(e.getMessage());
    }

    @Test
    void testMaxDepthZero() {
        // maxDepth(0) throws IllegalArgumentException when setting the option
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.ParseOptions.defaults().maxDepth(0));
        assertTrue(e.getMessage().contains("Maximum parsing depth must be positive"));
    }

    @Test
    void testMaxDepthNegative() {
        // maxDepth(-1) throws IllegalArgumentException when setting the option
        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.ParseOptions.defaults().maxDepth(-1));
        assertTrue(e.getMessage().contains("Maximum parsing depth must be positive"));
    }

    // ===== Line/Column Tracking Edge Cases =====

    @Test
    void testErrorLineColumnMultiline() {
        String json =
                "{\n"
                        + "  \"key1\": \"value1\",\n"
                        + "  \"key2\": ,\n"
                        + // Error here: missing value after colon
                        "  \"key3\": \"value3\"\n"
                        + "}";

        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertEquals(3, e.getLine()); // Error on line 3
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorLineColumnWithTabs() {
        String json = "{\t\"key\":\t}"; // Error: missing value after colon
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertEquals(1, e.getLine());
        // Column should account for tab as 1 character (not expanded)
        assertTrue(e.getColumn() > 0);
    }

    @Test
    void testErrorLineColumnInArray() {
        String json = "[1, 2, , 4]"; // Error: extra comma
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(json));
        assertEquals(1, e.getLine());
        // Column of extra comma (after "[1, 2, ")
        assertTrue(e.getColumn() >= 7);
    }

    // ===== Stringify Edge Cases =====

    @Test
    void testStringifyNull() {
        assertEquals("null", Json.stringify(null, false));
        assertEquals("null", Json.stringify(Json.NULL, false));
    }

    @Test
    void testStringifyBoolean() {
        assertEquals("true", Json.stringify(true, false));
        assertEquals("false", Json.stringify(false, false));
    }

    @Test
    void testStringifyBigInteger() {
        BigInteger big = new BigInteger("123456789012345678901234567890");
        assertEquals("123456789012345678901234567890", Json.stringify(big, false));
    }

    @Test
    void testStringifyBigDecimalScientificNotation() {
        // BigDecimal with scientific notation should use toPlainString()
        BigDecimal bd = new BigDecimal("1.23E+10");
        assertEquals("12300000000", Json.stringify(bd, false));

        bd = new BigDecimal("1.23E-10");
        assertEquals("0.000000000123", Json.stringify(bd, false));
    }

    @Test
    void testStringifyPrettyPrintingComplex() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("array", Arrays.asList(1L, 2L, 3L)); // Use Long
        obj.put("nested", Map.of("key", "value"));
        obj.put("number", 42L); // Use Long

        String pretty = Json.stringify(obj, true);
        assertTrue(pretty.contains("\n"));
        assertTrue(pretty.contains("  ")); // Indentation
        // Should be valid JSON
        Object parsed = Json.parse(pretty);
        // Compare content
        assertInstanceOf(Map.class, parsed);
        Map<?, ?> parsedMap = (Map<?, ?>) parsed;
        assertEquals(obj.size(), parsedMap.size());
        for (Map.Entry<String, Object> entry : obj.entrySet()) {
            Object parsedValue = parsedMap.get(entry.getKey());
            if (entry.getValue() instanceof List) {
                List<?> expectedList = (List<?>) entry.getValue();
                List<?> actualList = (List<?>) parsedValue;
                assertEquals(expectedList.size(), actualList.size());
                for (int i = 0; i < expectedList.size(); i++) {
                    assertEquals(expectedList.get(i), actualList.get(i));
                }
            } else {
                assertEquals(entry.getValue(), parsedValue);
            }
        }
    }

    // ===== ParseOptions Edge Cases =====

    @Test
    void testParseOptionsChaining() {
        Json.ParseOptions options =
                Json.ParseOptions.defaults()
                        .decimalsAsBigDecimal(true)
                        .maxDepth(500)
                        .decimalsAsBigDecimal(false); // Should override BigDecimal

        Object parsed = Json.parse("3.14", options);
        assertInstanceOf(Double.class, parsed);
        assertEquals(3.14, (Double) parsed, 0.001);
    }

    @Test
    void testParseOptionsDefaultIsBigDecimal() {
        // Default should be BigDecimal for floats
        Object parsed = Json.parse("3.14");
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("3.14"), parsed);
    }

    // ===== Round Trip Edge Cases =====

    @Test
    void testRoundTripAllTypes() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("null", Json.NULL);
        original.put("boolean", true);
        original.put("number", 42L);
        original.put("float", new BigDecimal("3.14"));
        original.put("string", "test");
        original.put("array", Arrays.asList(1L, 2L, 3L)); // Use Long instead of Integer
        original.put("object", Map.of("nested", "value"));
        original.put("emptyArray", Collections.emptyList());
        original.put("emptyObject", Collections.emptyMap());

        String json = Json.stringify(original, false);
        Object parsed = Json.parse(json);
        // Compare content
        assertInstanceOf(Map.class, parsed);
        Map<?, ?> parsedMap = (Map<?, ?>) parsed;
        assertEquals(original.size(), parsedMap.size());
        for (Map.Entry<String, Object> entry : original.entrySet()) {
            Object parsedValue = parsedMap.get(entry.getKey());
            if (entry.getValue() instanceof List) {
                // Compare list contents
                List<?> expectedList = (List<?>) entry.getValue();
                List<?> actualList = (List<?>) parsedValue;
                assertEquals(expectedList.size(), actualList.size());
                for (int i = 0; i < expectedList.size(); i++) {
                    assertEquals(expectedList.get(i), actualList.get(i));
                }
            } else if (entry.getValue() instanceof Map) {
                // Compare map contents
                Map<?, ?> expectedMap = (Map<?, ?>) entry.getValue();
                Map<?, ?> actualMap = (Map<?, ?>) parsedValue;
                assertEquals(expectedMap, actualMap);
            } else if (entry.getValue() instanceof BigDecimal) {
                // BigDecimal comparison
                assertEquals(
                        0, ((BigDecimal) entry.getValue()).compareTo((BigDecimal) parsedValue));
            } else {
                assertEquals(entry.getValue(), parsedValue);
            }
        }
    }

    @Test
    void testRoundTripWithUnicode() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("ascii", "Hello");
        original.put("unicode", "世界");
        original.put("emoji", "😀");
        original.put("mixed", "Hello 世界 😀");

        String json = Json.stringify(original, false);
        Object parsed = Json.parse(json);
        assertEquals(original, parsed);
    }

    @Test
    void testRoundTripLargeNumbers() {
        Map<String, Object> original = new LinkedHashMap<>();
        original.put("long", Long.MAX_VALUE);
        original.put("bigint", new BigInteger("123456789012345678901234567890"));
        original.put("bigdecimal", new BigDecimal("1234567890.12345678901234567890"));

        String json = Json.stringify(original, false);
        Object parsed = Json.parse(json, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));

        // Compare values
        Map<?, ?> parsedMap = (Map<?, ?>) parsed;
        assertEquals(Long.MAX_VALUE, parsedMap.get("long"));
        assertEquals(new BigInteger("123456789012345678901234567890"), parsedMap.get("bigint"));
        assertEquals(
                0,
                new BigDecimal("1234567890.12345678901234567890")
                        .compareTo((BigDecimal) parsedMap.get("bigdecimal")));
    }
}
