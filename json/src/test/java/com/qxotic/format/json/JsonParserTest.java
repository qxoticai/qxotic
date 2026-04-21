package com.qxotic.format.json;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

import java.util.List;
import java.util.Map;
import java.util.stream.Stream;

import static com.qxotic.format.json.TestUtils.list;
import static com.qxotic.format.json.TestUtils.map;
import static org.junit.jupiter.api.Assertions.*;

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
}
