package com.qxotic.format.json;

import static com.qxotic.format.json.TestUtils.list;
import static com.qxotic.format.json.TestUtils.map;
import static org.junit.jupiter.api.Assertions.*;

import java.math.BigDecimal;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/** Parsing of literals, structures, typed roots and malformed input. */
class JsonParserTest {

    @Test
    void testNullLiteral() {
        assertSame(Json.NULL, Json.parse("null"));
    }

    @Test
    void testBooleanLiterals() {
        assertEquals(true, Json.parse("true"));
        assertEquals(false, Json.parse("false"));
    }

    @Test
    void testEmptyStructures() {
        assertEquals("", Json.parse("\"\""));
        assertEquals(list(), Json.parse("[]"));
        assertEquals(map(), Json.parse("{}"));
    }

    @Test
    void testSimpleArray() {
        assertEquals(list(1L, "two", true, Json.NULL), Json.parse("[1,\"two\",true,null]"));
    }

    @Test
    void testSimpleObject() {
        assertEquals(
                map("name", "John", "age", 30L, "active", true),
                Json.parse("{\"name\":\"John\",\"age\":30,\"active\":true}"));
    }

    @Test
    void testNestedStructures() {
        // Nested arrays
        assertEquals(list(list(1L, 2L), list(3L, 4L)), Json.parse("[[1,2],[3,4]]"));

        // Nested objects
        assertEquals(map("outer", map("value", 42L)), Json.parse("{\"outer\":{\"value\":42}}"));

        // Mixed nesting
        assertEquals(
                list(1L, map("key", "value"), list()), Json.parse("[1,{\"key\":\"value\"},[]]"));
    }

    @Test
    void testWhitespaceHandling() {
        assertEquals(map("key", "value"), Json.parse("  {  \"key\"  :  \"value\"  }  "));

        String withNewlines = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        assertEquals(map("a", 1L, "b", 2L), Json.parse(withNewlines));
    }

    @ParameterizedTest
    @MethodSource("typedParseCases")
    void testTypedParsing(String json, Class<?> type, Object expected, String errorMessage) {
        if (expected instanceof Class && Throwable.class.isAssignableFrom((Class<?>) expected)) {
            @SuppressWarnings("unchecked")
            Class<? extends Throwable> exType = (Class<? extends Throwable>) expected;
            Throwable e = assertThrows(exType, () -> parseWithType(json, type));
            if (errorMessage != null) {
                assertTrue(e.getMessage().contains(errorMessage));
            }
        } else {
            Object result = parseWithType(json, type);
            if (expected instanceof Number) {
                assertEquals(
                        ((Number) expected).doubleValue(), ((Number) result).doubleValue(), 0.001);
            } else {
                assertEquals(expected, result);
            }
        }
    }

    static Stream<Arguments> typedParseCases() {
        return Stream.of(
                // Map parsing
                Arguments.of("{\"a\":1}", Map.class, map("a", 1L), null),
                Arguments.of(
                        "[1,2,3]",
                        Map.class,
                        Json.ParseException.class,
                        "Expected JSON object at root"),

                // List parsing
                Arguments.of("[1,2,3]", List.class, list(1L, 2L, 3L), null),
                Arguments.of(
                        "{\"a\":1}",
                        List.class,
                        Json.ParseException.class,
                        "Expected JSON array at root"),

                // String parsing
                Arguments.of("\"hello\"", String.class, "hello", null),
                Arguments.of(
                        "123",
                        String.class,
                        Json.ParseException.class,
                        "Expected JSON string at root"),

                // Number parsing
                Arguments.of("123", Number.class, 123L, null),
                Arguments.of("3.14", Number.class, 3.14, null),
                Arguments.of(
                        "true",
                        Number.class,
                        Json.ParseException.class,
                        "Expected JSON number at root"));
    }

    private static Object parseWithType(String json, Class<?> type) {
        if (type == Map.class) return Json.parseMap(json);
        if (type == List.class) return Json.parseList(json);
        if (type == String.class) return Json.parseString(json);
        if (type == Number.class) return Json.parseNumber(json);
        throw new IllegalArgumentException("Unknown type: " + type);
    }

    @Test
    void testNullInputRejection() {
        assertThrows(NullPointerException.class, () -> Json.parse((CharSequence) null));
        assertThrows(NullPointerException.class, () -> Json.isValid((CharSequence) null));
    }

    @Test
    void testIsValid() {
        assertTrue(Json.isValid("{\"a\":1}"));
        assertTrue(Json.isValid("[1,2,3]"));
        assertFalse(Json.isValid("{\"a\":}"));
    }

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
    }

    // ===== Null.toString() coverage =====

    @Test
    void testNullToString() {
        assertEquals("null", Json.NULL.toString());
    }

    // ===== isValid() with valid and invalid JSON =====

    @Test
    void testIsValidWithValidJson() {
        assertTrue(Json.isValid("{}"));
        assertTrue(Json.isValid("[]"));
        assertTrue(Json.isValid("true"));
        assertTrue(Json.isValid("false"));
        assertTrue(Json.isValid("null"));
        assertTrue(Json.isValid("123"));
        assertTrue(Json.isValid("\"string\""));
    }

    @Test
    void testIsValidWithInvalidJson() {
        assertFalse(Json.isValid("{"));
        assertFalse(Json.isValid("}"));
        assertFalse(Json.isValid(""));
        assertFalse(Json.isValid("undefined"));
    }

    @Test
    void testIsValidWithOptions() {
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(5);
        assertTrue(Json.isValid("[[[[[]]]]]", options));
        assertFalse(Json.isValid("[[[[[[]]]]]]", options)); // Depth 6 > 5
    }

    // ===== skipValue() uncovered branches =====

    @Test
    void testSkipValueWithClosingBracket() {
        // This triggers the ']' case in skipValue switch
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("]"));
        assertTrue(
                e.getMessage().contains("Expected value") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testSkipValueWithUnexpectedCharacter() {
        // This triggers the default case with non-digit, non-minus character
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("@"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    // ===== skipStringSimple() uncovered lines =====

    @Test
    void testSkipStringSimpleUnterminated() {
        // String that ends without closing quote
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"unterminated"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testSkipStringSimpleWithControlCharacter() {
        // String with unescaped control character (use actual control char in string)
        String jsonWithControl = "\"test" + (char) 0x01 + "value\"";
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(jsonWithControl));
        assertTrue(e.getMessage().contains("Control") || e.getMessage().contains("control"));
    }

    // ===== skipEscapeSequence() uncovered branches =====

    @Test
    void testInvalidEscapeSequence() {
        // Invalid escape character
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\z\""));
        assertTrue(e.getMessage().contains("Invalid escape"));
    }

    @Test
    void testEscapeSequenceWithUnicode() {
        // Valid unicode escape
        assertEquals("A", Json.parse("\"\\u0041\""));
    }

    // ===== scanNumber() '+' sign error =====

    @Test
    void testNumberWithPlusSign() {
        // JSON does not allow + sign before numbers
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("+123"));
        assertTrue(
                e.getMessage().contains("Unexpected '+'")
                        || e.getMessage().contains("Unexpected character"));
    }

    // ===== escapeFor() control characters =====

    @Test
    void testStringifyWithBackspace() {
        assertEquals("\"\\b\"", Json.stringify("\b", false));
    }

    @Test
    void testStringifyWithFormFeed() {
        assertEquals("\"\\f\"", Json.stringify("\f", false));
    }

    @Test
    void testStringifyWithCarriageReturn() {
        assertEquals("\"\\r\"", Json.stringify("\r", false));
    }

    @Test
    void testStringifyWithControlCharacters() {
        // Control characters (0x00-0x1F) need unicode escape per RFC 8259
        assertEquals("\"\\u0000\"", Json.stringify("\u0000", false));
        assertEquals("\"\\u001F\"", Json.stringify("\u001f", false)); // uppercase hex
        // DEL (0x7F) is NOT a control character per RFC 8259; not escaped
        assertEquals("\"\u007f\"", Json.stringify("\u007f", false));
    }

    // ===== print() unsupported type error =====

    @Test
    void testStringifyUnsupportedType() {
        class CustomClass {
            @Override
            public String toString() {
                return "custom";
            }
        }

        IllegalArgumentException e =
                assertThrows(
                        IllegalArgumentException.class,
                        () -> Json.stringify(new CustomClass(), false));
        assertTrue(e.getMessage().contains("Cannot serialize"));
    }

    // ===== parseTyped methods with wrong root type =====

    @Test
    void testParseMapWithArrayRoot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parseMap("[]"));
        assertTrue(e.getMessage().contains("Expected JSON object"));
    }

    @Test
    void testParseListWithObjectRoot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parseList("{}"));
        assertTrue(e.getMessage().contains("Expected JSON array"));
    }

    @Test
    void testParseStringWithNumberRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseString("123"));
        assertTrue(e.getMessage().contains("Expected JSON string"));
    }

    @Test
    void testParseNumberWithStringRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseNumber("\"123\""));
        assertTrue(e.getMessage().contains("Expected JSON number"));
    }

    @Test
    void testParseBooleanWithStringRoot() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parseBoolean("\"true\""));
        assertTrue(e.getMessage().contains("Expected JSON boolean"));
    }

    // ===== Additional edge cases for coverage =====

    @Test
    void testEmptyInput() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(""));
        assertTrue(
                e.getMessage().contains("Unexpected end")
                        || e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testWhitespaceOnlyInput() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("   \n\t  "));
        assertTrue(
                e.getMessage().contains("Unexpected end")
                        || e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testTrailingContent() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("{} {}"));
        assertTrue(e.getMessage().contains("Expected end of input"));
    }

    @Test
    void testArrayTrailingComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("[1, 2,]"));
        assertTrue(e.getMessage().contains("Expected value"));
    }

    @Test
    void testObjectTrailingComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\": 1,}"));
        assertTrue(e.getMessage().contains("Expected"));
    }

    @Test
    void testNumberStartingWithDot() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse(".123"));
        assertTrue(e.getMessage().contains("Unexpected character"));
    }

    @Test
    void testNumberWithLeadingZeros() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("0123"));
        assertTrue(e.getMessage().contains("Leading zeros"));
    }

    @Test
    void testNumberWithEmptyExponent() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        assertTrue(
                e.getMessage().contains("Exponent missing digits")
                        || e.getMessage().contains("Unexpected end"));
    }

    @Test
    void testNumberWithDecimalPointOnly() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("1."));
        assertTrue(e.getMessage().contains("Expected digit after decimal"));
    }

    @Test
    void testUnicodeEscapeInvalidHex() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00GH\""));
        assertTrue(e.getMessage().contains("Invalid hex digit"));
    }

    @Test
    void testUnicodeEscapeIncomplete() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"\\u00\""));
        assertTrue(
                e.getMessage().contains("Incomplete Unicode escape")
                        || e.getMessage().contains("Unexpected end"));
    }

    @Test
    void testNestedObjectWithDuplicateKeys() {
        Json.ParseOptions options = Json.ParseOptions.defaults().failOnDuplicateKeys(true);
        Json.ParseException e =
                assertThrows(
                        Json.ParseException.class,
                        () -> Json.parse("{\"a\": 1, \"a\": 2}", options));
        assertTrue(e.getMessage().contains("Duplicate key"));
    }

    @Test
    void testDeeplyNestedArray() {
        // Test that depth limit is enforced
        Json.ParseOptions options = Json.ParseOptions.defaults().maxDepth(5);
        StringBuilder sb = new StringBuilder();
        sb.append("[".repeat(6));
        sb.append("null");
        sb.append("]".repeat(6));

        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse(sb.toString(), options));
        assertTrue(e.getMessage().contains("Maximum parsing depth exceeded"));
    }

    @Test
    void testPrettyPrintWithDeepIndent() {
        // Test indent cache fallback for deep nesting
        Map<String, Object> nested = new HashMap<>();
        Map<String, Object> current = nested;

        // Create deeply nested structure
        for (int i = 0; i < 20; i++) {
            Map<String, Object> next = new HashMap<>();
            current.put("level" + i, next);
            current = next;
        }
        current.put("value", "deep");

        String pretty = Json.stringify(nested, true);
        assertTrue(pretty.contains("\n"));
        // Verify it's valid JSON
        Object parsed = Json.parse(pretty);
        assertNotNull(parsed);
    }

    @Test
    void testParseOptionsConfiguration() {
        // ParseOptions doesn't implement equals/hashCode - each instance is unique
        // Test configuration changes instead
        Json.ParseOptions opts1 = Json.ParseOptions.defaults();
        assertEquals(1000, opts1.maxDepth());
        assertTrue(opts1.decimalsAsBigDecimal());
        assertFalse(opts1.failOnDuplicateKeys());

        Json.ParseOptions opts2 = Json.ParseOptions.defaults().maxDepth(500);
        assertEquals(500, opts2.maxDepth());

        Json.ParseOptions opts3 = Json.ParseOptions.defaults().decimalsAsBigDecimal(false);
        assertFalse(opts3.decimalsAsBigDecimal());

        Json.ParseOptions opts4 = Json.ParseOptions.defaults().failOnDuplicateKeys(true);
        assertTrue(opts4.failOnDuplicateKeys());
    }

    @Test
    void testParseExceptionGetters() {
        try {
            Json.parse("{");
            fail("Should have thrown ParseException");
        } catch (Json.ParseException e) {
            // Just verify we can access the getters without exception
            // The actual values depend on the error context
            String msg = e.getMessage();
            int pos = e.getPosition();
            int line = e.getLine();
            int col = e.getColumn();

            // Verify message is present
            assertNotNull(msg);
            assertFalse(msg.isEmpty());

            // Verify getters return values (even if -1 for uninitialized)
            assertTrue(pos >= -1);
            assertTrue(line >= -1);
            assertTrue(col >= -1);
        }
    }

    @Test
    void testParseExceptionSimpleConstructor() {
        // Test simple constructors
        Json.ParseException e1 = new Json.ParseException("message");
        assertEquals("message", e1.getMessage());
        assertEquals(-1, e1.getPosition());

        Exception cause = new RuntimeException("cause");
        Json.ParseException e2 = new Json.ParseException("message", cause);
        assertEquals("message", e2.getMessage());
        assertEquals(cause, e2.getCause());
    }

    @Test
    void testDeepNesting() {
        int depth = 50;
        StringBuilder sb = new StringBuilder();
        sb.append("{\"a\":".repeat(depth));
        sb.append("1");
        sb.append("}".repeat(depth));
        Object result = Json.parse(sb.toString());
        for (int i = 0; i < depth; i++) {
            assertInstanceOf(Map.class, result);
            result = ((Map<?, ?>) result).get("a");
        }
        assertEquals(1L, result);
    }

    @Test
    void testWideNestedArrays() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append(i);
        }
        sb.append("]");
        List<?> result = (List<?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
        for (int i = 0; i < 50; i++) {
            assertEquals((long) i, result.get(i));
        }
    }

    @Test
    void testWideNestedObjects() {
        StringBuilder sb = new StringBuilder("{");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append("\"key").append(i).append("\":").append(i);
        }
        sb.append("}");
        Map<?, ?> result = (Map<?, ?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
        for (int i = 0; i < 50; i++) {
            assertEquals((long) i, result.get("key" + i));
        }
    }

    @Test
    void testLargeString() {
        StringBuilder sb = new StringBuilder("\"");
        sb.append("x".repeat(1000));
        sb.append("\"");
        String result = (String) Json.parse(sb.toString());
        assertEquals(1000, result.length());
    }

    @Test
    void testVeryLargeNumber() {
        StringBuilder sb = new StringBuilder("1");
        sb.append("0".repeat(500));
        Object result = Json.parse(sb.toString());
        assertTrue(result instanceof BigInteger || result instanceof BigDecimal);
    }

    @Test
    void testManyDecimals() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < 50; i++) {
            if (i > 0) sb.append(",");
            sb.append("0.").append(String.format("%03d", i));
        }
        sb.append("]");
        List<?> result = (List<?>) Json.parse(sb.toString());
        assertEquals(50, result.size());
    }

    @Test
    void testComplexMixedNesting() {
        String json = "{\"a\":[1,2,{\"b\":[3,4,{\"c\":5}]},6],\"d\":{\"e\":{\"f\":7}}}";
        Object result = Json.parse(json);
        assertInstanceOf(Map.class, result);
    }

    @Test
    void testStringWithEscapedQuotes() {
        String input = "\"He said, \\\"Hello World!\\\"\"";
        String result = (String) Json.parse(input);
        assertEquals("He said, \"Hello World!\"", result);
    }

    @Test
    void testStringWithMultipleEscapes() {
        String input = "\"Line1\\nLine2\\tTab\\rReturn\\\\Backslash\\/Slash\"";
        String result = (String) Json.parse(input);
        assertEquals("Line1\nLine2\tTab\rReturn\\Backslash/Slash", result);
    }

    @Test
    void testUnicodeEscapesInSequence() {
        String input = "\"\\u0041\\u0042\\u0043\\u0044\\u0045\"";
        String result = (String) Json.parse(input);
        assertEquals("ABCDE", result);
    }

    @Test
    void testArrayWithNulls() {
        String json = "[null,1,null,\"str\",null,[],null,{}]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(8, result.size());
        assertSame(Json.NULL, result.get(0));
        assertEquals(1L, result.get(1));
        assertSame(Json.NULL, result.get(2));
        assertEquals("str", result.get(3));
        assertSame(Json.NULL, result.get(4));
        assertInstanceOf(List.class, result.get(5));
        assertSame(Json.NULL, result.get(6));
        assertInstanceOf(Map.class, result.get(7));
    }

    @Test
    void testObjectWithNulls() {
        String json = "{\"a\":null,\"b\":1,\"c\":null,\"d\":\"str\"}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(4, result.size());
        assertSame(Json.NULL, result.get("a"));
        assertEquals(1L, result.get("b"));
        assertSame(Json.NULL, result.get("c"));
        assertEquals("str", result.get("d"));
    }

    @Test
    void testBooleanMixedWithNull() {
        String json = "[true,false,null,true,false,null]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
        assertEquals(true, result.get(0));
        assertEquals(false, result.get(1));
        assertSame(Json.NULL, result.get(2));
        assertEquals(true, result.get(3));
        assertEquals(false, result.get(4));
        assertSame(Json.NULL, result.get(5));
    }

    @Test
    void testWhitespaceVariations() {
        String json = "  \t\n  {  \t\n  \"key\"  \t\n  :  \t\n  \"value\"  \t\n  }  \t\n  ";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals("value", result.get("key"));
    }

    @Test
    void testNumbersWithExponents() {
        String json = "[1e0,1E1,1e-1,1E+1,1e10,1E-10]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
        for (Object o : result) {
            assertInstanceOf(BigDecimal.class, o);
        }
    }

    @Test
    void testNumbersWithDecimals() {
        String json = "[0.0,0.1,1.0,10.0,0.01,1.23,123.456]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(7, result.size());
        for (Object o : result) {
            assertInstanceOf(BigDecimal.class, o);
        }
    }

    @Test
    void testNegativeNumbers() {
        String json = "[-0,-1,-10,-0.5,-1e10,-123.456]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
    }

    @Test
    void testNumberZeroVariations() {
        String json = "[0,-0,0.0,-0.0,0e0,0E0]";
        List<?> result = (List<?>) Json.parse(json);
        assertEquals(6, result.size());
    }

    @Test
    void testEmptyKey() {
        String json = "{\"\": \"empty key\"}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(1, result.size());
        assertEquals("empty key", result.get(""));
    }

    @Test
    void testSpecialCharsInKey() {
        String json = "{\"a-b_c.d/e\": 1}";
        Map<?, ?> result = (Map<?, ?>) Json.parse(json);
        assertEquals(1, result.size());
        assertEquals(1L, result.get("a-b_c.d/e"));
    }

    @Test
    void testMaxSafeInteger() {
        long maxSafe = 9007199254740991L;
        String json = String.valueOf(maxSafe);
        Object result = Json.parse(json);
        assertEquals(maxSafe, result);
    }

    @Test
    void testMinSafeInteger() {
        long minSafe = -9007199254740991L;
        String json = String.valueOf(minSafe);
        Object result = Json.parse(json);
        assertEquals(minSafe, result);
    }

    @Test
    void testStringifiedNumber() {
        String json = "12345678901234567890";
        Object result = Json.parse(json);
        assertInstanceOf(BigInteger.class, result);
        assertEquals(new BigInteger("12345678901234567890"), result);
    }

    @Test
    void testPreciseDecimal() {
        String json = "0.123456789012345678901234567890";
        Object result = Json.parse(json);
        assertInstanceOf(BigDecimal.class, result);
    }

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
        // parseBigDecimal() returns the BigDecimal as-is, preserving original precision
        Object parsed =
                Json.parse("1.500", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("1.500"), parsed);

        parsed = Json.parse("1.500e2", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        assertEquals(new BigDecimal("1.500e2"), parsed);
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
        // stringify(null) now throws IllegalArgumentException; use Json.NULL for JSON null
        assertThrows(IllegalArgumentException.class, () -> Json.stringify(null, false));
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

    @Test
    void testRejectIncompleteLiterals() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("tru"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("fals"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("nul"));

        assertTrue(e1.getMessage().contains("Expected"));
        assertTrue(e2.getMessage().contains("Expected"));
        assertTrue(e3.getMessage().contains("Expected"));
    }

    @Test
    void testRejectLiteralWithSuffix() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("truefalse"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectArrayWithOnlyComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[, ]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectObjectWithOnlyComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("{,}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInArray() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[,1]"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectLeadingCommaInObject() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{,\"a\":1}"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingValueAfterComma() {
        Json.ParseException e = assertThrows(Json.ParseException.class, () -> Json.parse("[1,2,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectMissingMemberAfterComma() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1,"));
        assertTrue(e.getMessage().contains("Expected") || e.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectExponentWithoutDigits() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("1e"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("1e+"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("1e-"));

        assertTrue(e1.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e2.getMessage().toLowerCase().contains("exponent"));
        assertTrue(e3.getMessage().toLowerCase().contains("exponent"));
    }

    @Test
    void testRejectPlusSignAndBareNumberTokens() {
        Json.ParseException plus = assertThrows(Json.ParseException.class, () -> Json.parse("+1"));
        Json.ParseException minus = assertThrows(Json.ParseException.class, () -> Json.parse("-"));
        Json.ParseException bareDot =
                assertThrows(Json.ParseException.class, () -> Json.parse(".5"));

        assertTrue(
                plus.getMessage().contains("Unexpected") || plus.getMessage().contains("Expected"));
        assertTrue(minus.getMessage().contains("Expected"));
        assertTrue(
                bareDot.getMessage().contains("Unexpected")
                        || bareDot.getMessage().contains("Expected"));
    }

    @Test
    void testRejectUpperCaseLiterals() {
        Json.ParseException e1 = assertThrows(Json.ParseException.class, () -> Json.parse("TRUE"));
        Json.ParseException e2 = assertThrows(Json.ParseException.class, () -> Json.parse("FALSE"));
        Json.ParseException e3 = assertThrows(Json.ParseException.class, () -> Json.parse("NULL"));

        assertTrue(e1.getMessage().contains("Unexpected"));
        assertTrue(e2.getMessage().contains("Unexpected"));
        assertTrue(e3.getMessage().contains("Unexpected"));
    }

    @Test
    void testRejectTrailingGarbageAfterValidJson() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("{\"a\":1}\n  x"));
        assertTrue(e.getMessage().contains("end of input"));
    }

    @Test
    void testRejectUnterminatedEscapeAtEndOfString() {
        Json.ParseException e =
                assertThrows(Json.ParseException.class, () -> Json.parse("\"abc\\\""));
        assertTrue(e.getMessage().contains("Unexpected end of input"));
    }

    @Test
    void testRejectNonStringObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(1, "one");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Json.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }

    @Test
    void testRejectNullObjectKeyWhenStringifying() {
        Map<Object, Object> map = new LinkedHashMap<>();
        map.put(null, "value");

        IllegalArgumentException e =
                assertThrows(IllegalArgumentException.class, () -> Json.stringify(map, false));
        assertTrue(e.getMessage().contains("keys must be strings"));
    }

    @Test
    void testLargeNumbers() {
        BigInteger bigInt = new BigInteger("9999999999999999999999999999999999999999");
        Object parsed = Json.parse("9999999999999999999999999999999999999999");
        assertEquals(bigInt, parsed);
        assertEquals("9999999999999999999999999999999999999999", Json.stringify(parsed));

        // Test large decimal with BigDecimal mode
        String largeDec = "12345678901234567890.12345678901234567890";
        Object bdParsed =
                Json.parse(largeDec, Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, bdParsed);
        assertEquals(new BigDecimal(largeDec), bdParsed);
    }

    @Test
    void testUnicode() {
        String emoji = "\"\\uD83D\\uDE00\""; // 😀
        Object parsed = Json.parse(emoji);
        assertEquals("\uD83D\uDE00", parsed);
        String stringified = Json.stringify(parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"😀\"", stringified);

        String chinese = "\"\\u4E2D\\u6587\""; // 中文
        parsed = Json.parse(chinese);
        assertEquals("\u4e2d\u6587", parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"中文\"", Json.stringify(parsed));

        String mixed = "\"Hello \\u4E16\\u754C!\""; // Hello 世界!
        parsed = Json.parse(mixed);
        assertEquals("Hello \u4e16\u754c!", parsed);
        // JSON spec allows direct Unicode output
        assertEquals("\"Hello 世界!\"", Json.stringify(parsed));
    }

    @Test
    void testPrettyPrinting() {
        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("name", "John");
        obj.put("age", 30);
        obj.put("active", true);

        String compact = Json.stringify(obj, false);
        assertEquals("{\"name\":\"John\",\"age\":30,\"active\":true}", compact);

        String pretty = Json.stringify(obj, true);
        assertEquals(
                "{\n"
                        + "  \"name\": \"John\",\n"
                        + "  \"age\": 30,\n"
                        + "  \"active\": true\n"
                        + "}",
                pretty);

        List<Object> arr = new ArrayList<>();
        arr.add(1);
        arr.add(2);
        arr.add(3);

        String compactArr = Json.stringify(arr, false);
        assertEquals("[1,2,3]", compactArr);

        String prettyArr = Json.stringify(arr, true);
        assertEquals("[\n" + "  1,\n" + "  2,\n" + "  3\n" + "]", prettyArr);
    }

    @Test
    void testNullHandling() {
        assertSame(Json.NULL, Json.parse("null"));
        assertEquals("null", Json.stringify(Json.NULL));

        Map<String, Object> obj = new LinkedHashMap<>();
        obj.put("key1", Json.NULL);
        obj.put("key2", "value");
        String json = Json.stringify(obj);
        assertEquals("{\"key1\":null,\"key2\":\"value\"}", json);

        Object parsed = Json.parse(json);
        assertEquals(obj, parsed);
    }

    @Test
    void testWhitespace() {
        String json = "  {  \"key\"  :  \n  \"value\"  \t  }  ";
        Object parsed = Json.parse(json);
        assertEquals(Map.of("key", "value"), parsed);

        json = "  [  1  ,  2  ,  3  ]  ";
        parsed = Json.parse(json);
        assertEquals(List.of(1L, 2L, 3L), parsed);

        json = "  \"  hello  world  \"  ";
        parsed = Json.parse(json);
        assertEquals("  hello  world  ", parsed);
    }

    @Test
    void testDuplicateKeys() {
        String json = "{\"key\":\"value1\",\"key\":\"value2\"}";
        Map<String, Object> obj = (Map<String, Object>) Json.parse(json);
        assertEquals(1, obj.size());
        assertEquals("value2", obj.get("key"));
    }

    @Test
    void testNegativeZero() {
        // Test with BigDecimal mode
        Object parsed = Json.parse("-0", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        // -0 as integer parses to 0L (no negative zero for integers)
        assertEquals(0L, parsed);
        String stringified = Json.stringify(parsed);
        assertEquals("0", stringified);

        parsed = Json.parse("-0.0", Json.ParseOptions.defaults().decimalsAsBigDecimal(true));
        assertInstanceOf(BigDecimal.class, parsed);
        stringified = Json.stringify(parsed);
        // stripTrailingZeros() removes .0
        assertEquals("0", stringified);
    }
}
