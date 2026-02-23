package com.qxotic.format.json;

import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.Optional;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/** Comprehensive tests for the new query API and type check methods. */
class JSONQueryTest {

    // ============== Query String Tests ==============

    @Test
    @DisplayName("queryString: simple key access")
    void testQueryStringSimple() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = JSON.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("queryString: nested key access")
    void testQueryStringNested() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {\"profile\": {\"name\": \"Bob\"}}}");
        Optional<String> result = JSON.queryString(data, "user", "profile", "name");
        assertTrue(result.isPresent());
        assertEquals("Bob", result.get());
    }

    @Test
    @DisplayName("queryString: missing key returns empty")
    void testQueryStringMissing() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = JSON.queryString(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: type mismatch returns empty")
    void testQueryStringTypeMismatch() {
        Map<String, Object> data = JSON.parseMap("{\"age\": 30}");
        Optional<String> result = JSON.queryString(data, "age");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: intermediate not map returns empty")
    void testQueryStringIntermediateNotMap() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = JSON.queryString(data, "name", "invalid");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryString: with default value")
    void testQueryStringWithDefault() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        String result = JSON.queryString(data, "nonexistent").orElse("Default");
        assertEquals("Default", result);
    }

    @Test
    @DisplayName("queryString: empty string value")
    void testQueryStringEmpty() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"\"}");
        Optional<String> result = JSON.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("", result.get());
    }

    @Test
    @DisplayName("queryString: unicode string")
    void testQueryStringUnicode() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"日本語\"}");
        Optional<String> result = JSON.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("日本語", result.get());
    }

    // ============== Query Map Tests ==============

    @Test
    @DisplayName("queryMap: simple map access")
    void testQueryMapSimple() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {\"name\": \"Alice\"}}");
        Optional<Map<String, Object>> result = JSON.queryMap(data, "user");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get().get("name"));
    }

    @Test
    @DisplayName("queryMap: nested map access")
    void testQueryMapNested() {
        Map<String, Object> data = JSON.parseMap("{\"a\": {\"b\": {\"c\": 1}}}");
        Optional<Map<String, Object>> result = JSON.queryMap(data, "a", "b");
        assertTrue(result.isPresent());
        assertEquals(1L, result.get().get("c"));
    }

    @Test
    @DisplayName("queryMap: missing key returns empty")
    void testQueryMapMissing() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {}}");
        Optional<Map<String, Object>> result = JSON.queryMap(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryMap: type mismatch (string instead of map)")
    void testQueryMapTypeMismatch() {
        Map<String, Object> data = JSON.parseMap("{\"user\": \"not a map\"}");
        Optional<Map<String, Object>> result = JSON.queryMap(data, "user");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryMap: empty map")
    void testQueryMapEmpty() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {}}");
        Optional<Map<String, Object>> result = JSON.queryMap(data, "user");
        assertTrue(result.isPresent());
        assertTrue(result.get().isEmpty());
    }

    // ============== Query List Tests ==============

    @Test
    @DisplayName("queryList: simple list access")
    void testQueryListSimple() {
        Map<String, Object> data = JSON.parseMap("{\"items\": [1, 2, 3]}");
        Optional<List<Object>> result = JSON.queryList(data, "items");
        assertTrue(result.isPresent());
        assertEquals(3, result.get().size());
    }

    @Test
    @DisplayName("queryList: nested list access")
    void testQueryListNested() {
        Map<String, Object> data = JSON.parseMap("{\"data\": {\"items\": [\"a\", \"b\"]}}");
        Optional<List<Object>> result = JSON.queryList(data, "data", "items");
        assertTrue(result.isPresent());
        assertEquals(2, result.get().size());
    }

    @Test
    @DisplayName("queryList: missing key returns empty")
    void testQueryListMissing() {
        Map<String, Object> data = JSON.parseMap("{\"items\": []}");
        Optional<List<Object>> result = JSON.queryList(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryList: type mismatch (string instead of list)")
    void testQueryListTypeMismatch() {
        Map<String, Object> data = JSON.parseMap("{\"items\": \"not a list\"}");
        Optional<List<Object>> result = JSON.queryList(data, "items");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryList: empty list")
    void testQueryListEmpty() {
        Map<String, Object> data = JSON.parseMap("{\"items\": []}");
        Optional<List<Object>> result = JSON.queryList(data, "items");
        assertTrue(result.isPresent());
        assertTrue(result.get().isEmpty());
    }

    @Test
    @DisplayName("queryList: list of objects")
    void testQueryListOfObjects() {
        Map<String, Object> data =
                JSON.parseMap("{\"users\": [{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]}");
        Optional<List<Object>> result = JSON.queryList(data, "users");
        assertTrue(result.isPresent());
        assertEquals(2, result.get().size());
    }

    // ============== Query Boolean Tests ==============

    @Test
    @DisplayName("queryBoolean: true value")
    void testQueryBooleanTrue() {
        Map<String, Object> data = JSON.parseMap("{\"active\": true}");
        Optional<Boolean> result = JSON.queryBoolean(data, "active");
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    @Test
    @DisplayName("queryBoolean: false value")
    void testQueryBooleanFalse() {
        Map<String, Object> data = JSON.parseMap("{\"active\": false}");
        Optional<Boolean> result = JSON.queryBoolean(data, "active");
        assertTrue(result.isPresent());
        assertFalse(result.get());
    }

    @Test
    @DisplayName("queryBoolean: missing key returns empty")
    void testQueryBooleanMissing() {
        Map<String, Object> data = JSON.parseMap("{\"active\": true}");
        Optional<Boolean> result = JSON.queryBoolean(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryBoolean: type mismatch (string instead of boolean)")
    void testQueryBooleanTypeMismatch() {
        Map<String, Object> data = JSON.parseMap("{\"active\": \"yes\"}");
        Optional<Boolean> result = JSON.queryBoolean(data, "active");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryBoolean: nested access")
    void testQueryBooleanNested() {
        Map<String, Object> data =
                JSON.parseMap("{\"user\": {\"settings\": {\"notifications\": true}}}");
        Optional<Boolean> result = JSON.queryBoolean(data, "user", "settings", "notifications");
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    // ============== Query Number Tests ==============

    @Test
    @DisplayName("queryNumber: integer value")
    void testQueryNumberInteger() {
        Map<String, Object> data = JSON.parseMap("{\"age\": 30}");
        Optional<Number> result = JSON.queryNumber(data, "age");
        assertTrue(result.isPresent());
        assertEquals(30, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: decimal value")
    void testQueryNumberDecimal() {
        Map<String, Object> data = JSON.parseMap("{\"price\": 19.99}");
        Optional<Number> result = JSON.queryNumber(data, "price");
        assertTrue(result.isPresent());
        assertEquals(19.99, result.get().doubleValue(), 0.001);
    }

    @Test
    @DisplayName("queryNumber: negative value")
    void testQueryNumberNegative() {
        Map<String, Object> data = JSON.parseMap("{\"temperature\": -5}");
        Optional<Number> result = JSON.queryNumber(data, "temperature");
        assertTrue(result.isPresent());
        assertEquals(-5, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: zero value")
    void testQueryNumberZero() {
        Map<String, Object> data = JSON.parseMap("{\"count\": 0}");
        Optional<Number> result = JSON.queryNumber(data, "count");
        assertTrue(result.isPresent());
        assertEquals(0, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: missing key returns empty")
    void testQueryNumberMissing() {
        Map<String, Object> data = JSON.parseMap("{\"age\": 30}");
        Optional<Number> result = JSON.queryNumber(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryNumber: type mismatch (string instead of number)")
    void testQueryNumberTypeMismatch() {
        Map<String, Object> data = JSON.parseMap("{\"age\": \"thirty\"}");
        Optional<Number> result = JSON.queryNumber(data, "age");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("queryNumber: nested access")
    void testQueryNumberNested() {
        Map<String, Object> data = JSON.parseMap("{\"product\": {\"details\": {\"stock\": 100}}}");
        Optional<Number> result = JSON.queryNumber(data, "product", "details", "stock");
        assertTrue(result.isPresent());
        assertEquals(100, result.get().intValue());
    }

    @Test
    @DisplayName("queryNumber: scientific notation")
    void testQueryNumberScientific() {
        Map<String, Object> data = JSON.parseMap("{\"value\": 1.5e10}");
        Optional<Number> result = JSON.queryNumber(data, "value");
        assertTrue(result.isPresent());
        assertEquals(1.5e10, result.get().doubleValue(), 0.001);
    }

    // ============== Query (Raw) Tests ==============

    @Test
    @DisplayName("query: returns String")
    void testQueryString() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<Object> result = JSON.query(data, "name");
        assertTrue(result.isPresent());
        assertTrue(result.get() instanceof String);
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("query: returns Map")
    void testQueryMap() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {\"name\": \"Alice\"}}");
        Optional<Object> result = JSON.query(data, "user");
        assertTrue(result.isPresent());
        assertTrue(result.get() instanceof Map);
    }

    @Test
    @DisplayName("query: returns List")
    void testQueryList() {
        Map<String, Object> data = JSON.parseMap("{\"items\": [1, 2, 3]}");
        Optional<Object> result = JSON.query(data, "items");
        assertTrue(result.isPresent());
        assertTrue(result.get() instanceof List);
    }

    @Test
    @DisplayName("query: returns JSON.NULL for explicit null")
    void testQueryExplicitNull() {
        Map<String, Object> data = JSON.parseMap("{\"value\": null}");
        Optional<Object> result = JSON.query(data, "value");
        assertTrue(result.isPresent());
        assertSame(JSON.NULL, result.get());
    }

    @Test
    @DisplayName("query: returns empty for missing key")
    void testQueryMissing() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<Object> result = JSON.query(data, "nonexistent");
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("query: nested access")
    void testQueryNested() {
        Map<String, Object> data = JSON.parseMap("{\"a\": {\"b\": {\"c\": 1}}}");
        Optional<Object> result = JSON.query(data, "a", "b", "c");
        assertTrue(result.isPresent());
        assertEquals(1L, result.get());
    }

    // ============== Zero Keys (Cast) Tests ==============

    @Test
    @DisplayName("queryString: 0 keys casts root to String")
    void testQueryStringZeroKeys() {
        Object root = "test string";
        Optional<String> result = JSON.queryString(root);
        assertTrue(result.isPresent());
        assertEquals("test string", result.get());
    }

    @Test
    @DisplayName("queryMap: 0 keys casts root to Map")
    void testQueryMapZeroKeys() {
        Map<String, Object> root = Map.of("key", "value");
        Optional<Map<String, Object>> result = JSON.queryMap(root);
        assertTrue(result.isPresent());
        assertEquals("value", result.get().get("key"));
    }

    @Test
    @DisplayName("queryList: 0 keys casts root to List")
    void testQueryListZeroKeys() {
        List<Object> root = List.of(1, 2, 3);
        Optional<List<Object>> result = JSON.queryList(root);
        assertTrue(result.isPresent());
        assertEquals(3, result.get().size());
    }

    @Test
    @DisplayName("queryBoolean: 0 keys casts root to Boolean")
    void testQueryBooleanZeroKeys() {
        Object root = true;
        Optional<Boolean> result = JSON.queryBoolean(root);
        assertTrue(result.isPresent());
        assertTrue(result.get());
    }

    @Test
    @DisplayName("queryNumber: 0 keys casts root to Number")
    void testQueryNumberZeroKeys() {
        Object root = 42;
        Optional<Number> result = JSON.queryNumber(root);
        assertTrue(result.isPresent());
        assertEquals(42, result.get().intValue());
    }

    @Test
    @DisplayName("query: 0 keys returns root as-is")
    void testQueryZeroKeys() {
        Map<String, Object> root = Map.of("key", "value");
        Optional<Object> result = JSON.query(root);
        assertTrue(result.isPresent());
        assertSame(root, result.get());
    }

    @Test
    @DisplayName("queryString: 0 keys with wrong type returns empty")
    void testQueryStringZeroKeysWrongType() {
        Object root = 42;
        Optional<String> result = JSON.queryString(root);
        assertTrue(result.isEmpty());
    }

    // ============== Type Check Method Tests ==============

    @Test
    @DisplayName("isMap: returns true for Map")
    void testIsMapTrue() {
        assertTrue(JSON.isMap(Map.of()));
    }

    @Test
    @DisplayName("isMap: returns false for non-Map")
    void testIsMapFalse() {
        assertFalse(JSON.isMap("string"));
        assertFalse(JSON.isMap(42));
        assertFalse(JSON.isMap(List.of()));
        assertFalse(JSON.isMap(true));
        assertFalse(JSON.isMap(null));
    }

    @Test
    @DisplayName("isList: returns true for List")
    void testIsListTrue() {
        assertTrue(JSON.isList(List.of()));
        assertTrue(JSON.isList(new ArrayList<>()));
    }

    @Test
    @DisplayName("isList: returns false for non-List")
    void testIsListFalse() {
        assertFalse(JSON.isList("string"));
        assertFalse(JSON.isList(42));
        assertFalse(JSON.isList(Map.of()));
        assertFalse(JSON.isList(true));
        assertFalse(JSON.isList(null));
    }

    @Test
    @DisplayName("isString: returns true for String")
    void testIsStringTrue() {
        assertTrue(JSON.isString("hello"));
        assertTrue(JSON.isString(""));
    }

    @Test
    @DisplayName("isString: returns false for non-String")
    void testIsStringFalse() {
        assertFalse(JSON.isString(42));
        assertFalse(JSON.isString(Map.of()));
        assertFalse(JSON.isString(List.of()));
        assertFalse(JSON.isString(true));
        assertFalse(JSON.isString(null));
    }

    @Test
    @DisplayName("isNumber: returns true for Number")
    void testIsNumberTrue() {
        assertTrue(JSON.isNumber(42));
        assertTrue(JSON.isNumber(3.14));
        assertTrue(JSON.isNumber(0L));
    }

    @Test
    @DisplayName("isNumber: returns false for non-Number")
    void testIsNumberFalse() {
        assertFalse(JSON.isNumber("string"));
        assertFalse(JSON.isNumber(Map.of()));
        assertFalse(JSON.isNumber(List.of()));
        assertFalse(JSON.isNumber(true));
        assertFalse(JSON.isNumber(null));
    }

    @Test
    @DisplayName("isBoolean: returns true for Boolean")
    void testIsBooleanTrue() {
        assertTrue(JSON.isBoolean(true));
        assertTrue(JSON.isBoolean(false));
    }

    @Test
    @DisplayName("isBoolean: returns false for non-Boolean")
    void testIsBooleanFalse() {
        assertFalse(JSON.isBoolean("true"));
        assertFalse(JSON.isBoolean(1));
        assertFalse(JSON.isBoolean(Map.of()));
        assertFalse(JSON.isBoolean(List.of()));
        assertFalse(JSON.isBoolean(null));
    }

    // ============== Renamed Parsing Method Tests ==============

    @Test
    @DisplayName("parseMap: parses JSON object")
    void testParseMap() {
        Map<String, Object> result = JSON.parseMap("{\"name\": \"Alice\", \"age\": 30}");
        assertNotNull(result);
        assertEquals("Alice", result.get("name"));
        assertEquals(30L, result.get("age"));
    }

    @Test
    @DisplayName("parseMap: parses empty object")
    void testParseMapEmpty() {
        Map<String, Object> result = JSON.parseMap("{}");
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("parseMap: with options")
    void testParseMapWithOptions() {
        Map<String, Object> result = JSON.parseMap("{\"value\": 1.5}", JSON.options());
        assertNotNull(result);
        assertTrue(result.get("value") instanceof Number);
    }

    @Test
    @DisplayName("parseList: parses JSON array")
    void testParseList() {
        List<Object> result = JSON.parseList("[1, 2, 3]");
        assertNotNull(result);
        assertEquals(3, result.size());
        assertEquals(1L, result.get(0));
        assertEquals(2L, result.get(1));
        assertEquals(3L, result.get(2));
    }

    @Test
    @DisplayName("parseList: parses empty array")
    void testParseListEmpty() {
        List<Object> result = JSON.parseList("[]");
        assertNotNull(result);
        assertTrue(result.isEmpty());
    }

    @Test
    @DisplayName("parseList: parses array of objects")
    void testParseListOfObjects() {
        List<Object> result = JSON.parseList("[{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]");
        assertNotNull(result);
        assertEquals(2, result.size());
        assertTrue(result.get(0) instanceof Map);
        assertEquals("Alice", ((Map<?, ?>) result.get(0)).get("name"));
    }

    @Test
    @DisplayName("parseList: with options")
    void testParseListWithOptions() {
        List<Object> result = JSON.parseList("[1, 2, 3]", JSON.options());
        assertNotNull(result);
        assertEquals(3, result.size());
    }

    // ============== Deep Nesting Tests ==============

    @Test
    @DisplayName("query: very deep nesting (10 levels)")
    void testDeepNesting() {
        StringBuilder json = new StringBuilder("{");
        for (int i = 0; i < 10; i++) {
            json.append("\"level").append(i).append("\": {");
        }
        json.append("\"value\": \"found\"");
        for (int i = 0; i < 10; i++) {
            json.append("}");
        }
        json.append("}");

        Map<String, Object> data = JSON.parseMap(json.toString());
        String[] keys = new String[11];
        for (int i = 0; i < 10; i++) {
            keys[i] = "level" + i;
        }
        keys[10] = "value";

        Optional<String> result = JSON.queryString(data, keys);
        assertTrue(result.isPresent());
        assertEquals("found", result.get());
    }

    // ============== Complex JSON Tests ==============

    @Test
    @DisplayName("query: complex nested structure")
    void testComplexStructure() {
        String json =
                "{"
                        + "\"company\": {"
                        + "    \"name\": \"Acme Corp\","
                        + "    \"employees\": ["
                        + "        {\"id\": 1, \"name\": \"Alice\", \"active\": true, \"salary\": 50000.50},"
                        + "        {\"id\": 2, \"name\": \"Bob\", \"active\": false, \"salary\": 60000.00}"
                        + "    ],"
                        + "    \"metadata\": null,"
                        + "    \"settings\": {\"debug\": true, \"version\": 1.5}"
                        + "}}";

        Map<String, Object> data = JSON.parseMap(json);

        // Query various types
        Optional<String> companyName = JSON.queryString(data, "company", "name");
        assertTrue(companyName.isPresent());
        assertEquals("Acme Corp", companyName.get());

        Optional<List<Object>> employees = JSON.queryList(data, "company", "employees");
        assertTrue(employees.isPresent());
        assertEquals(2, employees.get().size());

        Optional<Map<String, Object>> settings = JSON.queryMap(data, "company", "settings");
        assertTrue(settings.isPresent());

        Optional<Object> metadata = JSON.query(data, "company", "metadata");
        assertTrue(metadata.isPresent());
        assertSame(JSON.NULL, metadata.get());
    }

    // ============== Error Handling Tests ==============

    @Test
    @DisplayName("query: null root throws NullPointerException")
    void testNullRoot() {
        assertThrows(
                NullPointerException.class,
                () -> {
                    JSON.queryString(null, "key");
                });
    }

    @Test
    @DisplayName("query: null key throws NullPointerException")
    void testNullKey() {
        Map<String, Object> data = Map.of("key", "value");
        assertThrows(
                NullPointerException.class,
                () -> {
                    JSON.queryString(data, (String) null);
                });
    }

    @Test
    @DisplayName("query: null in varargs throws NullPointerException")
    void testNullInVarargs() {
        Map<String, Object> data = JSON.parseMap("{\"a\": {\"b\": 1}}");
        assertThrows(
                NullPointerException.class,
                () -> {
                    JSON.queryString(data, "a", null, "b");
                });
    }

    // ============== Integration with Existing API Tests ==============

    @Test
    @DisplayName("query works with parseMap")
    void testQueryWithParseMap() {
        Map<String, Object> data = JSON.parseMap("{\"name\": \"Alice\"}");
        Optional<String> result = JSON.queryString(data, "name");
        assertTrue(result.isPresent());
        assertEquals("Alice", result.get());
    }

    @Test
    @DisplayName("query works with parseList")
    void testQueryWithParseList() {
        List<Object> data = JSON.parseList("[{\"name\": \"Alice\"}]");
        assertEquals(1, data.size());
        assertTrue(data.get(0) instanceof Map);
    }

    @Test
    @DisplayName("type checks work with parsed data")
    void testTypeChecksWithParsedData() {
        Map<String, Object> data =
                JSON.parseMap(
                        "{\"str\": \"text\", \"num\": 42, \"bool\": true, \"obj\": {}, \"arr\": []}");

        assertTrue(JSON.isString(data.get("str")));
        assertTrue(JSON.isNumber(data.get("num")));
        assertTrue(JSON.isBoolean(data.get("bool")));
        assertTrue(JSON.isMap(data.get("obj")));
        assertTrue(JSON.isList(data.get("arr")));
    }

    // ============== Edge Case Tests ==============

    @Test
    @DisplayName("query: key with special characters in value")
    void testKeyWithSpecialCharacters() {
        Map<String, Object> data = JSON.parseMap("{\"message\": \"Hello\\nWorld\\t!\"}");
        Optional<String> result = JSON.queryString(data, "message");
        assertTrue(result.isPresent());
        assertEquals("Hello\nWorld\t!", result.get());
    }

    @Test
    @DisplayName("query: very long string value")
    void testVeryLongString() {
        String longValue = "a".repeat(10000);
        Map<String, Object> data = JSON.parseMap("{\"data\": \"" + longValue + "\"}");
        Optional<String> result = JSON.queryString(data, "data");
        assertTrue(result.isPresent());
        assertEquals(longValue, result.get());
    }

    @Test
    @DisplayName("query: large number")
    void testLargeNumber() {
        Map<String, Object> data =
                JSON.parseMap("{\"big\": 9223372036854775807}"); // Long.MAX_VALUE
        Optional<Number> result = JSON.queryNumber(data, "big");
        assertTrue(result.isPresent());
        assertEquals(9223372036854775807L, result.get().longValue());
    }

    @Test
    @DisplayName("query: multiple queries on same data")
    void testMultipleQueries() {
        Map<String, Object> data = JSON.parseMap("{\"a\": 1, \"b\": 2, \"c\": 3}");

        Optional<Number> a = JSON.queryNumber(data, "a");
        Optional<Number> b = JSON.queryNumber(data, "b");
        Optional<Number> c = JSON.queryNumber(data, "c");

        assertTrue(a.isPresent() && a.get().intValue() == 1);
        assertTrue(b.isPresent() && b.get().intValue() == 2);
        assertTrue(c.isPresent() && c.get().intValue() == 3);
    }

    @Test
    @DisplayName("query: chain with filter and map")
    void testQueryChain() {
        Map<String, Object> data = JSON.parseMap("{\"user\": {\"age\": 25}}");

        boolean isAdult =
                JSON.queryNumber(data, "user", "age")
                        .filter(age -> age.intValue() >= 18)
                        .isPresent();

        assertTrue(isAdult);
    }

    @Test
    @DisplayName("query: functional composition")
    void testFunctionalComposition() {
        Map<String, Object> data =
                JSON.parseMap("{\"users\": [{\"name\": \"Alice\"}, {\"name\": \"Bob\"}]}");

        List<String> names =
                JSON.queryList(data, "users")
                        .map(
                                list -> {
                                    List<String> result = new ArrayList<>();
                                    for (Object obj : list) {
                                        if (obj instanceof Map) {
                                            Optional<String> name =
                                                    JSON.queryString(
                                                            (Map<String, Object>) obj, "name");
                                            name.ifPresent(result::add);
                                        }
                                    }
                                    return result;
                                })
                        .orElse(List.of());

        assertEquals(List.of("Alice", "Bob"), names);
    }
}
