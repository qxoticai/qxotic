package com.qxotic.format.json;

import static com.qxotic.format.json.TestUtils.*;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.*;

class JSONParserTest {

    @Test
    void testNullLiteral() {
        assertSame(JSON.NULL, JSON.parse("null"));
    }

    @Test
    void testBooleanLiterals() {
        assertEquals(true, JSON.parse("true"));
        assertEquals(false, JSON.parse("false"));
    }

    @Test
    void testEmptyStructures() {
        assertEquals("", JSON.parse("\"\""));
        assertEquals(list(), JSON.parse("[]"));
        assertEquals(map(), JSON.parse("{}"));
    }

    @Test
    void testSimpleArray() {
        assertEquals(list(1L, "two", true, JSON.NULL), JSON.parse("[1,\"two\",true,null]"));
    }

    @Test
    void testSimpleObject() {
        assertEquals(
                map("name", "John", "age", 30L, "active", true),
                JSON.parse("{\"name\":\"John\",\"age\":30,\"active\":true}"));
    }

    @Test
    void testNestedStructures() {
        // Nested arrays
        assertEquals(list(list(1L, 2L), list(3L, 4L)), JSON.parse("[[1,2],[3,4]]"));

        // Nested objects
        assertEquals(map("outer", map("value", 42L)), JSON.parse("{\"outer\":{\"value\":42}}"));

        // Mixed nesting
        assertEquals(
                list(1L, map("key", "value"), list()), JSON.parse("[1,{\"key\":\"value\"},[]]"));
    }

    @Test
    void testWhitespaceHandling() {
        assertEquals(map("key", "value"), JSON.parse("  {  \"key\"  :  \"value\"  }  "));

        String withNewlines = "{\n  \"a\": 1,\n  \"b\": 2\n}";
        assertEquals(map("a", 1L, "b", 2L), JSON.parse(withNewlines));
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
                        JSON.ParseException.class,
                        "Expected JSON object at root"),

                // List parsing
                Arguments.of("[1,2,3]", List.class, list(1L, 2L, 3L), null),
                Arguments.of(
                        "{\"a\":1}",
                        List.class,
                        JSON.ParseException.class,
                        "Expected JSON array at root"),

                // String parsing
                Arguments.of("\"hello\"", String.class, "hello", null),
                Arguments.of(
                        "123",
                        String.class,
                        JSON.ParseException.class,
                        "Expected JSON string at root"),

                // Number parsing
                Arguments.of("123", Number.class, 123L, null),
                Arguments.of("3.14", Number.class, 3.14, null),
                Arguments.of(
                        "true",
                        Number.class,
                        JSON.ParseException.class,
                        "Expected JSON number at root"));
    }

    private static Object parseWithType(String json, Class<?> type) {
        if (type == Map.class) return JSON.parseMap(json);
        if (type == List.class) return JSON.parseList(json);
        if (type == String.class) return JSON.parseString(json);
        if (type == Number.class) return JSON.parseNumber(json);
        throw new IllegalArgumentException("Unknown type: " + type);
    }

    @Test
    void testNullInputRejection() {
        assertThrows(NullPointerException.class, () -> JSON.parse((CharSequence) null));
        assertThrows(NullPointerException.class, () -> JSON.isValid((CharSequence) null));
    }

    @Test
    void testIsValid() {
        assertTrue(JSON.isValid("{\"a\":1}"));
        assertTrue(JSON.isValid("[1,2,3]"));
        assertFalse(JSON.isValid("{\"a\":}"));
    }
}
